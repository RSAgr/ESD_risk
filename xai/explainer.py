"""
=============================================================
XAI Module — SHAP + Attention Visualisation  (Member 3)
=============================================================
Provides:
  1. SHAP feature importance (global + local)
  2. Attention map visualisation (which time steps mattered)
  3. Risk explanation in plain English
  4. Knowledge Graph context reasoning (rule-based)
=============================================================
"""

import os, sys, time
import numpy as np
import pandas as pd
OUTPUT_DIR = os.path.dirname(__file__)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FEATURE_NAMES = ["Humidity (%)", "Temperature (°C)", "E-Field (V/m)", "Contact Voltage (V)", "Movement (g)"]
RISK_LABELS   = ["Low", "Medium", "High"]
RISK_COLORS   = ["#2ecc71", "#f39c12", "#e74c3c"]

import shap
import torch
import numpy as np


class SHAPExplainer:
    """
    True SHAP feature attribution for the numerical sensor inputs.

    Categorical inputs are kept fixed for the explained sample.
    Returns:
        (T, n_features) SHAP values
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def explain(self, numerical, categoricals):
        self.model.eval()

        numerical = numerical.to(self.device)
        categoricals = {
            k: v.to(self.device)
            for k, v in categoricals.items()
        }

        # Wrapper because SHAP expects only tensor inputs
        class WrappedModel(torch.nn.Module):
            def __init__(self, model, cats):
                super().__init__()
                self.model = model
                self.cats = cats

            def forward(self, x):
                batch_size = x.size(0)
                cats = {}

                for name, cat_tensor in self.cats.items():
                    if cat_tensor.dim() == 0:
                        cat_tensor = cat_tensor.unsqueeze(0)

                    if cat_tensor.size(0) == batch_size:
                        cats[name] = cat_tensor
                    elif cat_tensor.size(0) == 1:
                        cats[name] = cat_tensor.expand(batch_size)
                    else:
                        raise RuntimeError(
                            f"Categorical batch for {name} has size "
                            f"{cat_tensor.size(0)}, expected 1 or {batch_size}."
                        )

                return self.model(x, cats)

        wrapped = WrappedModel(self.model, categoricals)

        # Background = zeros (fastest and simplest)
        background = torch.zeros_like(numerical)

        explainer = shap.GradientExplainer(
            wrapped,
            background
        )

        shap_values = explainer.shap_values(numerical)

        with torch.no_grad():
            pred = wrapped(numerical).argmax(dim=1).item()

        # Multi-class handling. Depending on SHAP version, GradientExplainer
        # returns either a list per class or one array with a final class axis.
        if isinstance(shap_values, list):
            values = shap_values[pred][0]
        else:
            values = np.asarray(shap_values)
            if values.shape[0] == numerical.size(0):
                values = values[0]
            if values.ndim == 3:
                values = values[..., pred]

        return values


    def global_importance(self, loader, n_batches=10):
        """
        Average |SHAP| value for each feature.
        """

        all_importance = []

        for i, (num, cats, _) in enumerate(loader):

            if i >= n_batches:
                break

            batch = min(4, num.size(0))

            for j in range(batch):

                shap_vals = self.explain(
                    num[j:j+1],
                    {k: v[j:j+1] for k, v in cats.items()}
                )

                all_importance.append(self._feature_importance(shap_vals))

        return np.mean(all_importance, axis=0)

    @staticmethod
    def _feature_importance(shap_vals):
        values = np.asarray(shap_vals)
        while values.ndim > 2:
            values = values.mean(axis=-1)
        return np.abs(values).mean(axis=0)

# ── 1. SHAP-style Feature Importance (gradient-based) ─────────
class GradientExplainer:
    """
    Gradient × Input saliency — lightweight SHAP proxy.
    Works without the shap library (no external dependency).
    For real SHAP use: shap.DeepExplainer(model, background)
    """
    def __init__(self, model, device):
        self.model  = model
        self.device = device

    def explain(self, numerical, categoricals):
        """
        Returns saliency scores: (T, n_features) — importance per timestep per feature.
        """
        self.model.eval()
        num = numerical.clone().requires_grad_(True).to(self.device)
        cats = {k: v.to(self.device) for k, v in categoricals.items()}

        logits = self.model(num, cats)
        pred_class = logits.argmax(dim=-1)

        # Backprop w.r.t. predicted class score
        score = logits[0, pred_class]
        score.backward()

        # Gradient × Input saliency
        saliency = (num.grad * num).squeeze(0).abs().detach().cpu().numpy()
        return saliency   # (T, n_features)

    def global_importance(self, loader, n_batches=10):
        """Average feature importance over multiple windows."""
        all_importance = []
        for i, (num, cats, _) in enumerate(loader):
            if i >= n_batches:
                break
            for j in range(min(4, num.size(0))):
                sal = self.explain(
                    num[j:j+1],
                    {k: v[j:j+1] for k, v in cats.items()}
                )
                all_importance.append(sal.mean(axis=0))  # mean over time → (n_features,)
        return np.mean(all_importance, axis=0)


# def plot_global_importance(importances, save=True):
#     """Bar chart of feature importances."""
#     fig, ax = plt.subplots(figsize=(8, 4))
#     colors = ["#3498db" if i < 3 else "#9b59b6" for i in range(len(FEATURE_NAMES))]
#     bars = ax.barh(FEATURE_NAMES, importances, color=colors, edgecolor="white", height=0.6)
#     ax.set_xlabel("Mean |Gradient × Input| Importance", fontsize=11)
#     ax.set_title("ESD Risk — Feature Importance (Global)", fontsize=13, fontweight="bold")
#     ax.spines[["top","right"]].set_visible(False)
#     for bar, val in zip(bars, importances):
#         ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
#                 f"{val:.3f}", va="center", fontsize=9)
#     plt.tight_layout()
#     path = os.path.join(OUTPUT_DIR, "global_importance.png")
#     if save:
#         plt.savefig(path, dpi=150, bbox_inches="tight")
#         print(f"  Saved: {path}")
#     plt.close()
#     return path

def plot_global_importance(
    importances,
    title="ESD Risk — Feature Importance (Global)",
    xlabel="Feature Importance",
    filename="global_importance.png",
    save=True,
):
    """Bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["#3498db" if i < 3 else "#9b59b6"
              for i in range(len(FEATURE_NAMES))]

    bars = ax.barh(
        FEATURE_NAMES,
        importances,
        color=colors,
        edgecolor="white",
        height=0.6,
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, importances):
        ax.text(
            val + 0.001,
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)

    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.close()
    return path


def compare_explainer_runtime(model, loader, device, n_batches=10):
    """
    Run GradientExplainer and SHAPExplainer on the same loader.

    Returns timing plus both importance vectors.
    """
    gradient_explainer = GradientExplainer(model, device)
    shap_explainer = SHAPExplainer(model, device)

    start = time.perf_counter()
    gradient_importance = gradient_explainer.global_importance(loader, n_batches=n_batches)
    gradient_seconds = time.perf_counter() - start

    start = time.perf_counter()
    shap_importance = shap_explainer.global_importance(loader, n_batches=n_batches)
    shap_seconds = time.perf_counter() - start

    speedup = shap_seconds / gradient_seconds if gradient_seconds > 0 else float("inf")

    return {
        "gradient_importance": gradient_importance,
        "shap_importance": shap_importance,
        "gradient_seconds": gradient_seconds,
        "shap_seconds": shap_seconds,
        "speedup": speedup,
    }


# ── 2. Attention Map Visualisation ────────────────────────────
def plot_attention_map(model, numerical, categoricals, device, save=True):
    """Heatmap of attention weights across time steps."""
    model.eval()
    num  = numerical.to(device)
    cats = {k: v.to(device) for k, v in categoricals.items()}

    with torch.no_grad():
        logits, attn_list = model(num, cats, return_attention=True)
        pred  = logits.argmax(dim=-1).item()
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    # Average attention across heads, use last layer
    attn = attn_list[-1].squeeze(0).cpu().numpy()   # (T, T)
    avg_attn = attn.mean(axis=0)   # (T,) — how much each position was attended to

    T = avg_attn.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 3]})

    # Top: attention weights over time
    axes[0].fill_between(range(T), avg_attn, alpha=0.7, color="#3498db")
    axes[0].set_ylabel("Attention\nWeight", fontsize=9)
    axes[0].set_xlim(0, T-1)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[0].set_title(
        f"Attention Map — Predicted Risk: {RISK_LABELS[pred]} "
        f"(conf {probs[pred]:.1%})", fontsize=12, fontweight="bold"
    )

    # Bottom: sensor signals
    num_np = numerical.squeeze(0).cpu().numpy()
    signal_labels = ["Humidity", "Temperature", "E-Field", "Voltage", "Movement"]
    signal_colors = ["#2ecc71", "#e67e22", "#e74c3c", "#9b59b6", "#3498db"]
    for i, (lbl, col) in enumerate(zip(signal_labels, signal_colors)):
        normed = (num_np[:, i] - num_np[:, i].min()) / (np.ptp(num_np[:, i]) + 1e-8)
        axes[1].plot(normed + i * 1.1, label=lbl, color=col, linewidth=1.2)
    axes[1].set_xlabel("Time steps", fontsize=10)
    axes[1].set_ylabel("Normalised sensor signals", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "attention_map.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()
    return path


# ── 3. Plain-English Risk Explanation ─────────────────────────
def generate_explanation(sensor_values: dict, risk_level: int, probs: np.ndarray) -> str:
    """
    Rule-augmented natural language explanation of a prediction.
    Combines model output with domain knowledge (KG rules).
    """
    h  = sensor_values.get("humidity_pct", 50)
    ef = sensor_values.get("efield_vm", 0)
    cv = sensor_values.get("contact_voltage_v", 0)
    mv = sensor_values.get("movement_g", 0)
    temp = sensor_values.get("temperature_c", 24)
    activity = sensor_values.get("activity", "unknown")
    env      = sensor_values.get("environment", "unknown")
    fabric   = sensor_values.get("fabric_type", "unknown")

    label = RISK_LABELS[risk_level]
    conf  = probs[risk_level] * 100

    lines = [
        f"⚡ ESD RISK PREDICTION: {label.upper()} ({conf:.1f}% confidence)",
        f"{'─'*50}",
        "",
        "📊 SENSOR ANALYSIS:"
    ]

    # Humidity
    if h < 30:
        lines.append(f"  • Humidity {h:.0f}% — CRITICAL LOW. Very dry conditions greatly increase static buildup.")
    elif h < 50:
        lines.append(f"  • Humidity {h:.0f}% — Low. Increased ESD susceptibility.")
    else:
        lines.append(f"  • Humidity {h:.0f}% — Normal. Helps dissipate static charge.")

    # Electric field
    if ef > 2000:
        lines.append(f"  • E-Field {ef:.0f} V/m — DANGER. High electrostatic field detected.")
    elif ef > 800:
        lines.append(f"  • E-Field {ef:.0f} V/m — Elevated. Monitor closely.")
    else:
        lines.append(f"  • E-Field {ef:.0f} V/m — Normal range.")

    # Contact voltage
    if cv > 500:
        lines.append(f"  • Contact voltage {cv:.0f} V — HIGH. Risk of discharge on touch.")
    elif cv > 200:
        lines.append(f"  • Contact voltage {cv:.0f} V — Moderate triboelectric buildup.")
    else:
        lines.append(f"  • Contact voltage {cv:.0f} V — Low.")

    # Movement
    if mv > 2.5:
        lines.append(f"  • Movement {mv:.2f} g — Active. High friction increases charge generation.")
    else:
        lines.append(f"  • Movement {mv:.2f} g — Minimal motion.")

    # Context (KG-based rules)
    lines += ["", "🔗 CONTEXT REASONING (Knowledge Graph):"]
    if fabric in ["polyester", "synthetic", "wool"]:
        lines.append(f"  • Fabric [{fabric}] is a HIGH triboelectric generator → elevated base risk.")
    else:
        lines.append(f"  • Fabric [{fabric}] has low triboelectric tendency.")

    if env in ["lab", "office"]:
        lines.append(f"  • Environment [{env}] likely has sensitive electronics nearby → higher impact risk.")
    else:
        lines.append(f"  • Environment [{env}] — lower electronic sensitivity.")

    if activity == "handling_electronics":
        lines.append(f"  • Activity [handling electronics] — DIRECT ESD risk to components.")
    elif activity == "running":
        lines.append(f"  • Activity [running] — Rapid movement increases charge buildup rate.")

    # Mitigation
    lines += ["", "🛡️ RECOMMENDED ACTIONS:"]
    if risk_level == 2:
        lines += [
            "  ✗ Do NOT touch sensitive electronics right now.",
            "  → Use ESD wrist strap immediately.",
            "  → Increase ambient humidity if possible.",
            "  → Touch grounded surface before handling components.",
        ]
    elif risk_level == 1:
        lines += [
            "  ⚠ Proceed with caution.",
            "  → Consider ESD protection before handling electronics.",
            "  → Avoid synthetic fabrics in this environment.",
        ]
    else:
        lines += [
            "  ✓ Conditions are currently safe.",
            "  → Continue monitoring — conditions can change rapidly.",
        ]

    return "\n".join(lines)


# ── 4. Knowledge Graph summary ────────────────────────────────
KNOWLEDGE_GRAPH = {
    "nodes": [
        {"id": "ESD_Risk",         "type": "concept"},
        {"id": "Humidity",         "type": "sensor"},
        {"id": "ElectricField",    "type": "sensor"},
        {"id": "ContactVoltage",   "type": "sensor"},
        {"id": "Movement",         "type": "sensor"},
        {"id": "Polyester",        "type": "material"},
        {"id": "Wool",             "type": "material"},
        {"id": "Synthetic",        "type": "material"},
        {"id": "Cotton",           "type": "material"},
        {"id": "Lab",              "type": "environment"},
        {"id": "Office",           "type": "environment"},
        {"id": "HandlingElectronics", "type": "activity"},
        {"id": "WristStrap",       "type": "mitigation"},
        {"id": "Grounding",        "type": "mitigation"},
        {"id": "HumidityControl",  "type": "mitigation"},
    ],
    "edges": [
        {"from": "Humidity",         "to": "ESD_Risk",    "relation": "inversely_affects", "weight": 0.35},
        {"from": "ElectricField",    "to": "ESD_Risk",    "relation": "directly_causes",   "weight": 0.30},
        {"from": "ContactVoltage",   "to": "ESD_Risk",    "relation": "directly_causes",   "weight": 0.25},
        {"from": "Movement",         "to": "ContactVoltage", "relation": "generates",      "weight": 0.20},
        {"from": "Polyester",        "to": "ContactVoltage", "relation": "amplifies",      "weight": 0.30},
        {"from": "Wool",             "to": "ContactVoltage", "relation": "amplifies",      "weight": 0.25},
        {"from": "Synthetic",        "to": "ContactVoltage", "relation": "amplifies",      "weight": 0.28},
        {"from": "Lab",              "to": "ESD_Risk",    "relation": "increases_impact",  "weight": 0.20},
        {"from": "Office",           "to": "ESD_Risk",    "relation": "increases_impact",  "weight": 0.15},
        {"from": "HandlingElectronics", "to": "ESD_Risk", "relation": "maximises_impact",  "weight": 0.25},
        {"from": "WristStrap",       "to": "ESD_Risk",    "relation": "mitigates",         "weight": -0.8},
        {"from": "Grounding",        "to": "ESD_Risk",    "relation": "mitigates",         "weight": -0.7},
        {"from": "HumidityControl",  "to": "Humidity",    "relation": "increases",         "weight": 0.6},
    ]
}

def get_knowledge_graph():
    return KNOWLEDGE_GRAPH


if __name__ == "__main__":
    print("XAI module loaded. Run via the main notebook.")
    kg = get_knowledge_graph()
    print(f"Knowledge Graph: {len(kg['nodes'])} nodes, {len(kg['edges'])} edges")
