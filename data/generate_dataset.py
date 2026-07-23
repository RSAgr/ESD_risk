"""
=============================================================
ESD Risk Dataset Generator  (Member 1 simulation)
=============================================================
Generates synthetic multi-modal wearable IoT sensor data
with realistic ESD risk patterns.

Sensors simulated:
  - Humidity (%)          : lower humidity → higher ESD risk
  - Temperature (°C)      : moderate effect
  - Electric Field (V/m)  : direct ESD precursor
  - Contact Voltage (V)   : triboelectric charge buildup
  - Body Movement (g)     : accelerometer — friction proxy

Risk Labels:
  0 = Low    (safe)
  1 = Medium (caution)
  2 = High   (danger)
=============================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ── reproducibility ──────────────────────────────────────────
np.random.seed(42)

# ── configuration ────────────────────────────────────────────
N_SAMPLES   = 15_000   # total time steps
SAMPLE_RATE = 50       # Hz  (50 samples/second)

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
elif "ROOT" in globals():
    PROJECT_ROOT = Path(ROOT).resolve()
else:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent
    while PROJECT_ROOT.parent != PROJECT_ROOT and not (PROJECT_ROOT / "data").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_PATH = PROJECT_ROOT / "data" / "esd_dataset.csv"


# ── helpers ──────────────────────────────────────────────────
def smooth(signal, sigma=5):
    return gaussian_filter1d(signal.astype(float), sigma=sigma)

def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ── generate base environmental signals ──────────────────────
t = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES)

# Humidity: daily sinusoidal pattern + noise  (high → safe)
humidity_base = 50 + 20 * np.sin(2 * np.pi * t / 300)
humidity_noise = np.random.normal(0, 3, N_SAMPLES)
humidity = clamp(smooth(humidity_base + humidity_noise, 10), 10, 95)

# Temperature: slow drift + noise
temp_base = 24 + 4 * np.sin(2 * np.pi * t / 600) + np.random.normal(0, 0.8, N_SAMPLES)
temperature = clamp(smooth(temp_base, 8), 15, 40)

# Body movement: bursty accelerometer (walking, sitting, running)
movement_base = np.zeros(N_SAMPLES)
for _ in range(80):
    start = np.random.randint(0, N_SAMPLES - 300)
    length = np.random.randint(50, 300)
    intensity = np.random.uniform(0.3, 3.0)
    movement_base[start:start+length] += intensity
movement = clamp(smooth(movement_base + np.random.exponential(0.1, N_SAMPLES), 4), 0, 5)

# Electric field: inversely related to humidity + movement-induced spikes
efield_base = 800 - 6 * humidity + 30 * movement + np.random.normal(0, 40, N_SAMPLES)
# Add sporadic high-field spikes (ESD precursor events)
spike_idx = np.random.choice(N_SAMPLES, size=120, replace=False)
efield_base[spike_idx] += np.random.uniform(500, 2000, size=120)
efield = clamp(smooth(efield_base, 3), 0, 5000)

# Contact voltage: triboelectric buildup  (friction × dryness)
dryness = clamp(100 - humidity, 5, 90) / 90.0
voltage_base = dryness * (200 + 80 * movement) + np.random.normal(0, 15, N_SAMPLES)
voltage_spike_idx = np.random.choice(N_SAMPLES, size=80, replace=False)
voltage_base[voltage_spike_idx] += np.random.uniform(200, 1000, size=80)
contact_voltage = clamp(smooth(voltage_base, 3), 0, 2000)


# ── compute ESD risk score (physics-inspired) ─────────────────
# Normalise inputs to [0, 1]
h_norm  = 1 - (humidity - 10) / 85          # low humidity → high risk
e_norm  = efield / 5000
v_norm  = contact_voltage / 2000
m_norm  = movement / 5
t_norm  = clamp((temperature - 15) / 25, 0, 1)

risk_score = (
    0.35 * h_norm +
    0.30 * e_norm +
    0.25 * v_norm +
    0.07 * m_norm +
    0.03 * t_norm
)
risk_score = clamp(smooth(risk_score, 2) + np.random.normal(0, 0.02, N_SAMPLES), 0, 1)

# Discretise into 3 risk classes
def score_to_label(s):
    if s < 0.35:   return 0   # Low
    elif s < 0.65: return 1   # Medium
    else:          return 2   # High

labels = np.array([score_to_label(s) for s in risk_score])

# Add context features (categorical, simulated)
activities = np.random.choice(
    ["sitting", "walking", "running", "handling_electronics"],
    size=N_SAMPLES,
    p=[0.45, 0.30, 0.10, 0.15]
)
environments = np.random.choice(
    ["office", "lab", "outdoor", "home"],
    size=N_SAMPLES,
    p=[0.35, 0.25, 0.20, 0.20]
)
fabric_types = np.random.choice(
    ["cotton", "polyester", "wool", "synthetic"],
    size=N_SAMPLES,
    p=[0.30, 0.35, 0.15, 0.20]
)

# Timestamps
timestamps = pd.date_range("2024-01-01", periods=N_SAMPLES, freq="20ms")

# ── assemble DataFrame ────────────────────────────────────────
df = pd.DataFrame({
    "timestamp":       timestamps,
    "humidity_pct":    np.round(humidity, 2),
    "temperature_c":   np.round(temperature, 2),
    "efield_vm":       np.round(efield, 2),
    "contact_voltage_v": np.round(contact_voltage, 2),
    "movement_g":      np.round(movement, 3),
    "activity":        activities,
    "environment":     environments,
    "fabric_type":     fabric_types,
    "risk_score":      np.round(risk_score, 4),
    "risk_label":      labels,
})

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# ── summary ───────────────────────────────────────────────────
counts = df["risk_label"].value_counts().sort_index()
print("=" * 50)
print("  ESD Dataset Generated Successfully")
print("=" * 50)
print(f"  Total samples : {len(df):,}")
print(f"  Output file   : {OUTPUT_PATH}")
print(f"\n  Risk distribution:")
labels_map = {0: "Low", 1: "Medium", 2: "High"}
for k, v in counts.items():
    print(f"    {labels_map[k]:8s} ({k}): {v:5,}  ({v/len(df)*100:.1f}%)")
print("=" * 50)

if __name__ == "__main__":
    pass
