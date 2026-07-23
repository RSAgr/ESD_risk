# ⚡ ESD Risk Prediction Framework
## Explainable Attention-Based Situational Awareness for Wearable IoT Devices

---

## Project Structure

```
esd_project/
├── data/
│   └── generate_dataset.py     # 🔵 Member 1 — synthetic ESD sensor data
├── models/
│   ├── attention_model.py      # 🟣 Member 2 — multi-head attention model
│   └── train.py                # 🟣 Member 2 — training + ablation + baselines
├── xai/
│   └── explainer.py            # 🔴 Member 3 — SHAP saliency + attention viz + KG
├── dashboard/
│   └── app.py                  # 🔴 Member 3 — real-time Streamlit dashboard
├── notebooks/
│   └── ESD_Full_Pipeline.ipynb # 📓 End-to-end pipeline (all members)
├── requirements.txt
└── README.md
```

---

## Quickstart

### Option A - Local
```bash
pip install -r requirements.txt

# Step 1: Generate dataset
python data/generate_dataset.py

# Step 2: Train model
python models/train.py

# Step 3: Launch dashboard
streamlit run dashboard/app.py
```

The scripts resolve paths relative to the project folder, so they can be run from the repo root without editing hardcoded Colab or machine-specific paths.

### Option B - Google Colab
1. Upload all `.py` files to `/content/esd_project/` in Colab
2. Open `notebooks/ESD_Full_Pipeline.ipynb`
3. Runtime -> Run All

---

## System Architecture

```
Wearable Sensors (Member 1)
  Humidity | Temperature | E-Field | Contact Voltage | Movement
       ↓
Data Pipeline
  Edge Preprocessing → Timestamped CSV Dataset
       ↓
Attention Model (Member 2)
  Input Encoder → Positional Encoding
  → Multi-Head Self-Attention (×2 layers, 4 heads)
  → Context Fusion (activity, environment, fabric embeddings)
  → Risk Classifier → Low / Medium / High
       ↓
XAI + Knowledge Graph (Member 3)
  Gradient Saliency → Feature Importance
  Attention Maps → Temporal Explanation
  KG Reasoning → Context-Aware Risk Paths
  Plain-English Explanation + Mitigation
       ↓
Real-Time Dashboard
  Live sensor streams | Risk gauge | Attention heatmap | Explanation
```

---

## Model Architecture

| Component           | Detail                              |
|---------------------|-------------------------------------|
| Input               | 5 numerical sensors, window=50 steps |
| Context             | 3 categorical embeddings (dim=8 each)|
| Encoder             | Linear projection → d_model=64      |
| Positional Encoding | Sinusoidal                          |
| Attention           | 2 × Multi-Head (4 heads, ff_dim=128)|
| Pooling             | Global average over time            |
| Fusion              | Concat numerical + categorical      |
| Classifier          | 64→3 softmax (Low/Medium/High)      |
| Parameters          | ~45,000                             |

---

## Dataset

| Property        | Value                          |
|-----------------|--------------------------------|
| Samples         | 15,000 time steps              |
| Sample Rate     | 50 Hz                          |
| Sensors         | Humidity, Temp, E-Field, Voltage, Movement |
| Context         | Activity, Environment, Fabric  |
| Labels          | 3-class ESD risk (Low/Med/High)|
| Generation      | Physics-inspired simulation    |

---

## Evaluation Metrics

- Accuracy, Macro F1, AUC-ROC
- Ablation study (heads, layers, context)
- Baseline comparison (Random Forest, SVM)
- Attention faithfulness (qualitative)
- Explanation quality (coverage, consistency)

---

## Team

| Member   | Responsibility                         | Thesis Chapters              |
|----------|----------------------------------------|------------------------------|
| Member 1 | IoT sensing, ESD physics, dataset      | System Architecture, Hardware |
| Member 2 | Attention model, training, ablation    | Methodology, Results         |
| Member 3 | XAI, Knowledge Graph, dashboard        | Explainability, Discussion   |
| All      | Integration, intro, conclusion         | Introduction, Conclusion     |
