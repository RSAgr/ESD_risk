"""
=============================================================
ESD Situational Awareness Dashboard
=============================================================
Real-time dashboard built with Streamlit.
Simulates live sensor streaming + model inference + XAI.

Run with:
    streamlit run dashboard/app.py
=============================================================
"""

import sys, os, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.attention_model import build_model
from xai.explainer import generate_explanation, get_knowledge_graph, FEATURE_NAMES, RISK_LABELS, RISK_COLORS

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ESD Risk Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-card { 
        border-radius: 12px; padding: 20px; text-align: center;
        font-size: 2.2em; font-weight: bold; color: white;
        margin-bottom: 10px;
    }
    .risk-LOW    { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .risk-MEDIUM { background: linear-gradient(135deg, #d35400, #f39c12); }
    .risk-HIGH   { background: linear-gradient(135deg, #922b21, #e74c3c); }
    .metric-box  {
        background: #1e1e2e; border-radius: 10px; padding: 15px;
        border: 1px solid #2d2d44; margin: 5px 0;
    }
    .stMetric label { color: #aaa !important; font-size: 0.85em !important; }
    div[data-testid="metric-container"] { background: #1a1a2e; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = build_model(window_size=50)
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, True
    return model, False


# ── sensor simulator ──────────────────────────────────────────
class SensorSimulator:
    """Generates realistic streaming sensor values."""
    def __init__(self, scenario="normal"):
        self.t       = 0
        self.scenario = scenario
        self._humidity     = np.random.uniform(40, 70)
        self._efield       = np.random.uniform(200, 600)
        self._voltage      = np.random.uniform(50, 200)
        self._movement     = 0.5
        self._temperature  = 23.0

    def step(self):
        self.t += 1
        dt = 0.1

        if self.scenario == "high_risk":
            self._humidity    = max(10, self._humidity - 0.3 + np.random.normal(0, 1))
            self._efield      = min(4500, self._efield + 15 + np.random.normal(0, 50))
            self._voltage     = min(1800, self._voltage + 10 + np.random.normal(0, 30))
            self._movement    = min(4.5, self._movement + 0.1 + np.random.uniform(0, 0.3))
        elif self.scenario == "recovery":
            self._humidity    = min(70, self._humidity + 0.5)
            self._efield      = max(100, self._efield - 20)
            self._voltage     = max(50, self._voltage - 15)
            self._movement    = max(0.1, self._movement - 0.05)
        else:
            self._humidity   += np.random.normal(0, 0.5)
            self._efield     += np.random.normal(0, 30)
            self._voltage    += np.random.normal(0, 15)
            self._movement    = max(0, self._movement + np.random.normal(0, 0.2))

        self._humidity    = np.clip(self._humidity, 10, 95)
        self._efield      = np.clip(self._efield, 0, 5000)
        self._voltage     = np.clip(self._voltage, 0, 2000)
        self._movement    = np.clip(self._movement, 0, 5)
        self._temperature += np.random.normal(0, 0.1)
        self._temperature  = np.clip(self._temperature, 18, 35)

        return {
            "humidity_pct":       round(self._humidity, 1),
            "temperature_c":      round(self._temperature, 1),
            "efield_vm":          round(self._efield, 1),
            "contact_voltage_v":  round(self._voltage, 1),
            "movement_g":         round(self._movement, 3),
        }


def sensors_to_tensor(history_buffer, scaler=None):
    """Convert 50-step sensor history to model input tensor."""
    arr = np.array([[
        h["humidity_pct"], h["temperature_c"], h["efield_vm"],
        h["contact_voltage_v"], h["movement_g"]
    ] for h in history_buffer[-50:]])

    if scaler:
        arr = scaler.transform(arr)
    else:
        # Simple normalisation using expected ranges
        mins  = [10, 15, 0, 0, 0]
        maxs  = [95, 40, 5000, 2000, 5]
        arr   = (arr - np.array(mins)) / (np.array(maxs) - np.array(mins) + 1e-8)

    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


def get_categoricals(activity, environment, fabric):
    act_map = {"sitting": 0, "walking": 1, "running": 2, "handling_electronics": 3}
    env_map = {"office": 0, "lab": 1, "outdoor": 2, "home": 3}
    fab_map = {"cotton": 0, "polyester": 1, "wool": 2, "synthetic": 3}
    return {
        "activity":    torch.tensor([act_map.get(activity, 0)], dtype=torch.long),
        "environment": torch.tensor([env_map.get(environment, 0)], dtype=torch.long),
        "fabric_type": torch.tensor([fab_map.get(fabric, 0)],  dtype=torch.long),
    }


# ── main app ──────────────────────────────────────────────────
KG_POSITIONS = {
    "ESD_Risk": (0.0, 0.0),
    "Humidity": (-1.7, 0.75),
    "ElectricField": (-1.7, 0.18),
    "ContactVoltage": (-1.7, -0.42),
    "Movement": (-3.05, -0.42),
    "Polyester": (-3.05, -1.0),
    "Wool": (-2.4, -1.28),
    "Synthetic": (-3.65, -1.28),
    "Cotton": (-3.7, -0.78),
    "Lab": (1.65, 0.62),
    "Office": (1.65, 0.08),
    "Outdoor": (2.55, 0.42),
    "Home": (2.55, -0.08),
    "Sitting": (3.0, -0.85),
    "Walking": (2.4, -1.15),
    "Running": (1.75, -1.05),
    "HandlingElectronics": (1.8, -0.52),
    "WristStrap": (0.15, -1.42),
    "Grounding": (0.95, -1.25),
    "HumidityControl": (-2.95, 1.05),
    "AntistaticMat": (-0.65, -1.62),
    "Ionizer": (-1.15, 1.35),
}

KG_TYPE_COLORS = {
    "concept": "#f1c40f",
    "sensor": "#3498db",
    "material": "#9b59b6",
    "environment": "#1abc9c",
    "activity": "#e67e22",
    "mitigation": "#2ecc71",
}


def active_knowledge_graph_paths(kg, reading, fabric, environment, activity):
    """Return active edge ids, active node ids, and human-readable path summaries."""
    active_edges = set()
    active_nodes = {"ESD_Risk"}
    paths = []

    def activate(source, target, label):
        active_edges.add((source, target))
        active_nodes.update([source, target])
        paths.append(label)

    h = reading["humidity_pct"]
    ef = reading["efield_vm"]
    cv = reading["contact_voltage_v"]

    if h < 40:
        activate("Humidity", "ESD_Risk", f"Low Humidity ({h}%) -> increases ESD Risk [weight: 0.35]")
    if ef > 1000:
        activate("ElectricField", "ESD_Risk", f"High E-Field ({ef:.0f} V/m) -> increases ESD Risk [weight: 0.30]")
    if cv > 300:
        activate("ContactVoltage", "ESD_Risk", f"High Contact Voltage ({cv:.0f} V) -> increases ESD Risk [weight: 0.25]")
    if fabric in ["polyester", "synthetic", "wool"]:
        fabric_node = {"polyester": "Polyester", "synthetic": "Synthetic", "wool": "Wool"}[fabric]
        activate(fabric_node, "ContactVoltage", f"{fabric.capitalize()} fabric -> amplifies Contact Voltage")
        active_edges.add(("ContactVoltage", "ESD_Risk"))
        active_nodes.update(["ContactVoltage", "ESD_Risk"])
    if environment in ["lab", "office", "home"]:
        env_node = {"lab": "Lab", "office": "Office", "home": "Home"}[environment]
        activate(env_node, "ESD_Risk", f"{environment.capitalize()} environment -> changes impact risk")
    if environment == "outdoor":
        activate("Outdoor", "Humidity", "Outdoor environment -> variable humidity")
        active_edges.add(("Humidity", "ESD_Risk"))
        active_nodes.update(["Humidity", "ESD_Risk"])
    if activity in ["sitting", "walking", "running"]:
        activity_node = {"sitting": "Sitting", "walking": "Walking", "running": "Running"}[activity]
        activate(activity_node, "Movement", f"{activity.replace('_', ' ').capitalize()} -> changes charge-generating movement")
        if activity in ["walking", "running"]:
            active_edges.add(("Movement", "ContactVoltage"))
            active_edges.add(("ContactVoltage", "ESD_Risk"))
            active_nodes.update(["ContactVoltage", "ESD_Risk"])
    if activity == "handling_electronics":
        activate("HandlingElectronics", "ESD_Risk", "Handling electronics -> maximises ESD impact")

    known_edges = {(edge["from"], edge["to"]) for edge in kg["edges"]}
    return active_edges & known_edges, active_nodes, paths


def build_knowledge_graph_figure(kg, active_edges=None, active_nodes=None):
    active_edges = active_edges or set()
    active_nodes = active_nodes or set()

    fig = go.Figure()

    for edge in kg["edges"]:
        source, target = edge["from"], edge["to"]
        x0, y0 = KG_POSITIONS[source]
        x1, y1 = KG_POSITIONS[target]
        is_active = (source, target) in active_edges
        color = "#2ecc71" if edge["weight"] < 0 else "#e74c3c"
        width = 4 if is_active else 1.5
        opacity = 0.95 if is_active else 0.35

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            hoverinfo="text",
            text=f"{source} -> {target}<br>{edge['relation']}<br>weight: {edge['weight']:.2f}",
            showlegend=False,
        ))
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=width,
            arrowcolor=color, opacity=opacity,
            standoff=18, startstandoff=18,
        )

    fig.add_trace(go.Scatter(
        x=[(KG_POSITIONS[e["from"]][0] + KG_POSITIONS[e["to"]][0]) / 2 for e in kg["edges"]],
        y=[(KG_POSITIONS[e["from"]][1] + KG_POSITIONS[e["to"]][1]) / 2 for e in kg["edges"]],
        mode="text",
        text=[edge["relation"].replace("_", " ") for edge in kg["edges"]],
        textfont=dict(color="#d8dee9", size=10),
        hoverinfo="skip",
        showlegend=False,
    ))

    node_x, node_y, node_text, node_type, node_color, node_size, node_line_width = [], [], [], [], [], [], []
    for node in kg["nodes"]:
        x, y = KG_POSITIONS[node["id"]]
        is_active = node["id"] in active_nodes
        node_x.append(x)
        node_y.append(y)
        node_text.append(node["id"].replace("_", " "))
        node_type.append(node["type"])
        node_color.append(KG_TYPE_COLORS.get(node["type"], "#95a5a6"))
        node_size.append(34 if node["id"] == "ESD_Risk" else 25 if is_active else 19)
        node_line_width.append(4 if is_active else 1.5)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        customdata=node_type,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(color="#ffffff", width=node_line_width),
        ),
        textfont=dict(color="#ffffff", size=12),
        hovertemplate="<b>%{text}</b><br>type: %{customdata}<extra></extra>",
        showlegend=False,
    ))

    for name, color in [
        ("Sensor", KG_TYPE_COLORS["sensor"]),
        ("Material", KG_TYPE_COLORS["material"]),
        ("Context", KG_TYPE_COLORS["environment"]),
        ("Activity", KG_TYPE_COLORS["activity"]),
        ("Mitigation", KG_TYPE_COLORS["mitigation"]),
        ("Risk", KG_TYPE_COLORS["concept"]),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=name,
            marker=dict(size=10, color=color),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(visible=False, range=[-4.05, 2.35]),
        yaxis=dict(visible=False, range=[-1.65, 1.28]),
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
    )
    return fig


def main():
    st.title("⚡ ESD Situational Awareness Dashboard")
    st.caption("Explainable Attention-Based ESD Risk Prediction for Wearable IoT Devices")

    model, model_loaded = load_model()
    if not model_loaded:
        st.warning("⚠️ No trained model found. Run training first, or predictions will use untrained weights.")

    # ── sidebar controls ──────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ Controls")
        scenario  = st.selectbox("Scenario", ["normal", "high_risk", "recovery"])
        activity  = st.selectbox("Activity", ["sitting", "walking", "running", "handling_electronics"])
        environ   = st.selectbox("Environment", ["office", "lab", "outdoor", "home"])
        fabric    = st.selectbox("Fabric type", ["cotton", "polyester", "wool", "synthetic"])
        auto_run  = st.toggle("Auto-refresh (live mode)", value=False)
        refresh_ms = st.slider("Refresh interval (ms)", 300, 2000, 800)
        st.divider()
        st.markdown("**About**\n\nThis dashboard simulates real-time ESD risk prediction using a multi-head attention model trained on synthetic wearable IoT sensor data.")

    # ── session state ─────────────────────────────────────────
    if "sim" not in st.session_state:
        st.session_state.sim     = SensorSimulator(scenario)
        st.session_state.history = []
        st.session_state.risk_history = []
        st.session_state.tick    = 0

    if st.session_state.sim.scenario != scenario:
        st.session_state.sim = SensorSimulator(scenario)

    # Step simulator
    reading = st.session_state.sim.step()
    st.session_state.history.append(reading)
    if len(st.session_state.history) > 300:
        st.session_state.history = st.session_state.history[-300:]
    st.session_state.tick += 1

    # ── inference ─────────────────────────────────────────────
    cats = get_categoricals(activity, environ, fabric)
    if len(st.session_state.history) >= 50:
        num_tensor = sensors_to_tensor(st.session_state.history)
        with torch.no_grad():
            logits, attn_maps = model(num_tensor, cats, return_attention=True)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        pred_class = int(probs.argmax())
        attn_last = attn_maps[-1].squeeze(0).mean(dim=0).detach().numpy()
    else:
        probs = np.array([0.8, 0.15, 0.05])
        pred_class = 0
        attn_last = np.ones(50) / 50
    
    st.session_state.risk_history.append(pred_class)
    if len(st.session_state.risk_history) > 300:
        st.session_state.risk_history = st.session_state.risk_history[-300:]

    # ── layout ────────────────────────────────────────────────
    col_risk, col_metrics = st.columns([1, 3])

    with col_risk:
        risk_name = RISK_LABELS[pred_class]
        st.markdown(
            f'<div class="risk-card risk-{risk_name}">⚡<br>{risk_name}<br>RISK</div>',
            unsafe_allow_html=True
        )
        st.metric("Low",    f"{probs[0]*100:.1f}%")
        st.metric("Medium", f"{probs[1]*100:.1f}%")
        st.metric("High",   f"{probs[2]*100:.1f}%")

    with col_metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Humidity",    f"{reading['humidity_pct']}%",
                  delta=f"{reading['humidity_pct']-50:.1f}")
        m2.metric("Temperature", f"{reading['temperature_c']}°C")
        m3.metric("E-Field",     f"{reading['efield_vm']:.0f} V/m")
        m4.metric("Voltage",     f"{reading['contact_voltage_v']:.0f} V")
        m5.metric("Movement",    f"{reading['movement_g']:.2f} g")

    st.divider()

    # ── charts ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sensor Streams", "🧠 Attention Map", "💡 Explanation", "🔗 Knowledge Graph"])

    with tab1:
        if len(st.session_state.history) > 1:
            hist_df = pd.DataFrame(st.session_state.history)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("Humidity & Temperature", "E-Field & Contact Voltage", "Movement & Risk"))
            x = list(range(len(hist_df)))

            fig.add_trace(go.Scatter(x=x, y=hist_df["humidity_pct"], name="Humidity", line=dict(color="#2ecc71")), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=hist_df["temperature_c"], name="Temp", line=dict(color="#e67e22"), yaxis="y2"), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=hist_df["efield_vm"], name="E-Field", line=dict(color="#e74c3c")), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=hist_df["contact_voltage_v"], name="Voltage", line=dict(color="#9b59b6")), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=hist_df["movement_g"], name="Movement", line=dict(color="#3498db")), row=3, col=1)

            if len(st.session_state.risk_history) > 0:
                risk_colors_map = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
                risk_numeric = st.session_state.risk_history
                fig.add_trace(go.Scatter(
                    x=list(range(len(risk_numeric))),
                    y=risk_numeric,
                    name="Risk Level",
                    mode="lines",
                    line=dict(color="#f1c40f", width=2),
                ), row=3, col=1)

            fig.update_layout(height=500, template="plotly_dark", showlegend=True,
                              margin=dict(l=40, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if len(st.session_state.history) >= 50:
            fig_attn = go.Figure()
            fig_attn.add_trace(go.Bar(
                x=list(range(len(attn_last))),
                y=attn_last,
                marker_color=attn_last,
                marker_colorscale="Viridis",
                name="Attention Weight",
            ))
            fig_attn.update_layout(
                title="Temporal Attention Weights (last 50 time steps)",
                xaxis_title="Time Step",
                yaxis_title="Attention Weight",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig_attn, use_container_width=True)

            # Feature importance proxy (gradient magnitudes)
            num_np = np.array([[
                h["humidity_pct"], h["temperature_c"], h["efield_vm"],
                h["contact_voltage_v"], h["movement_g"]
            ] for h in st.session_state.history[-50:]])
            # Simple variance-based proxy for feature importance
            importance = np.std(num_np, axis=0)
            importance = importance / importance.sum()
            fig_imp = go.Figure(go.Bar(
                x=FEATURE_NAMES, y=importance,
                marker_color=["#3498db", "#e67e22", "#e74c3c", "#9b59b6", "#2ecc71"]
            ))
            fig_imp.update_layout(
                title="Feature Contribution (signal variance proxy)",
                template="plotly_dark", height=300,
                yaxis_title="Relative Importance"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Collecting sensor data... need 50 steps for attention analysis.")

    with tab3:
        explanation = generate_explanation(reading, pred_class, probs)
        st.code(explanation, language=None)

        # Risk probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probs[2] * 100,
            title={"text": "High Risk Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS[pred_class]},
                "steps": [
                    {"range": [0, 35],  "color": "#1a4a2a"},
                    {"range": [35, 65], "color": "#4a3000"},
                    {"range": [65, 100],"color": "#4a0000"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 65}
            }
        ))
        fig_gauge.update_layout(height=280, template="plotly_dark", margin=dict(t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with tab4:
        kg = get_knowledge_graph()
        # Build adjacency info
        st.subheader("Knowledge Graph — ESD Risk Reasoning")
        st.caption(f"{len(kg['nodes'])} entities · {len(kg['edges'])} relationships")

        # Display edges as table
        active_edges, active_nodes, paths = active_knowledge_graph_paths(
            kg, reading, fabric, environ, activity
        )
        fig_kg = build_knowledge_graph_figure(kg, active_edges, active_nodes)
        st.plotly_chart(fig_kg, use_container_width=True)

        edge_df = pd.DataFrame(kg["edges"])
        edge_df["weight"] = edge_df["weight"].apply(lambda w: f"{'↑' if w > 0 else '↓'} {abs(w):.2f}")
        st.dataframe(edge_df, use_container_width=True, hide_index=True)

        # Active risk paths
        st.subheader("🔴 Active Risk Paths")
        if paths:
            for p in paths:
                st.error(f"⚠ {p}")
        else:
            st.success("✓ No high-risk paths active in current conditions.")

    # ── auto refresh ──────────────────────────────────────────
    if auto_run:
        time.sleep(refresh_ms / 1000)
        st.rerun()
    else:
        if st.button("⟳ Step (manual)", use_container_width=True):
            st.rerun()


if __name__ == "__main__":
    main()
