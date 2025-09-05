
# Customer Satisfaction Prediction (End-to-End)

An end-to-end machine learning project to predict **Customer Satisfaction Rating (1–5)** from customer support ticket data.
Includes modular Python package, training & inference scripts, starter notebooks, and a Streamlit app.

## ✨ Features
- Clean, modular `src/` package with sklearn `Pipeline` + `ColumnTransformer` (numerical, categorical, and text).
- Handles missing values, time parsing, class imbalance (optional), and basic NLP (TF–IDF on ticket text).
- Reproducible training via `train.py`, evaluation via `evaluate.py`, and batch predictions via `predict.py`.
- Streamlit app (`app.py`) for EDA, training (optional), and live predictions.
- Starter Jupyter notebooks for EDA and modeling experiments.
- Clear config in `config/config.yaml` and sensible `.gitignore` that **does not block your CSV**.

## 📂 Structure
```
customer_satisfaction_prediction/
├─ app.py
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ Makefile
├─ config/
│  └─ config.yaml
├─ data/
│  ├─ README.md
│  ├─ sample_customer_support_tickets_small.csv
│  └─ .gitkeep
├─ models/
│  ├─ .gitkeep
│  └─ label_encoder.joblib
├─ notebooks/
│  ├─ 01_eda.ipynb
│  └─ 02_modeling.ipynb
├─ reports/
│  └─ .gitkeep
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_loading.py
│  ├─ preprocessing.py
│  ├─ features.py
│  ├─ nlp_text.py
│  ├─ modeling.py
│  ├─ evaluate.py
│  ├─ utils.py
│  └─ visualization.py
├─ train.py
└─ predict.py
```

## 🚀 Quickstart

1) **Create environment & install deps**
```bash
pip install -r requirements.txt
```

2) **Add your dataset**
- Put your CSV at: `data/customer_support_tickets.csv`
- Or update the path in `config/config.yaml`

3) **Train**
```bash
python train.py
```
Artifacts saved to `models/`.

4) **Evaluate**
```bash
python evaluate.py
```

5) **Predict** (batch; outputs `predictions.csv`)
```bash
python predict.py --input data/customer_support_tickets.csv --output predictions.csv
```

6) **Run Streamlit app**
```bash
streamlit run app.py
```

## 🧾 Expected columns (from your PDF)
- Ticket ID, Customer Name, Customer Email, Customer Age, Customer Gender,
  Product Purchased, Date of Purchase, Ticket Type, Ticket Subject,
  Ticket Description, Ticket Status, Resolution, Ticket Priority,
  Ticket Channel, First Response Time, Time to Resolution,
  Customer Satisfaction Rating

If some columns are missing, the pipeline will ignore them gracefully when possible.

## 🧰 Config overview
See `config/config.yaml` to adjust file paths, model choice, and text column names.

## 🧪 Notes
- A tiny synthetic sample CSV is included so everything runs out-of-the-box.
- Replace it with your real dataset when ready.
