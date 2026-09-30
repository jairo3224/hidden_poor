# ============================================================
# NOTEBOOK 04 — EVALUATION
# Compares all three models on key metrics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD TEST DATA AND MODELS ─────────────────────────────
print("Loading test data and models...")
X_test = pd.read_csv('../data/X_test.csv')
y_test = pd.read_csv('../data/y_test.csv').squeeze()

scaler = joblib.load('../outputs/models/scaler.pkl')
lr     = joblib.load('../outputs/models/logistic_regression.pkl')
cb     = CatBoostClassifier(); cb.load_model('../outputs/models/catboost_model.cbm')
xgb    = joblib.load('../outputs/models/xgboost_model.pkl')

X_test_scaled = scaler.transform(X_test)

# ── 2. GET PREDICTIONS ───────────────────────────────────────
models = {
    'Logistic Regression\n(Baseline)': (lr, X_test_scaled),
    'CatBoost': (cb, X_test),
    'XGBoost':  (xgb, X_test),
}

results = []
for name, (model, X) in models.items():
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Core metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    # Exclusion error = false negatives / total actual poor
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    excl_err = fn / (fn + tp) if (fn + tp) > 0 else 0
    incl_err = fp / (fp + tn) if (fp + tn) > 0 else 0

    results.append({
        'Model': name,
        'Accuracy':         round(acc, 4),
        'Precision':        round(prec, 4),
        'Recall':           round(rec, 4),
        'F1-Score':         round(f1, 4),
        'AUC-ROC':          round(auc, 4),
        'Exclusion Error':  round(excl_err, 4),
        'Inclusion Error':  round(incl_err, 4),
        'True Poor Caught': tp,
        'Hidden Poor (FN)': fn,
    })

    print(f"\n{'='*50}")
    print(f"MODEL: {name.strip()}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Non-Poor','Poor']))
    print(f"AUC-ROC:         {auc:.4f}")
    print(f"Exclusion Error: {excl_err:.4f} ({fn} poor households misclassified as non-poor)")
    print(f"Inclusion Error: {incl_err:.4f}")

# ── 3. RESULTS TABLE ─────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("FULL COMPARISON TABLE")
print("="*70)
print(results_df.to_string(index=False))
results_df.to_csv('../outputs/results/comparison_table.csv', index=False)
print("\nSaved: comparison_table.csv")

# ── 4. VISUALIZATIONS ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison\n(Higher = Better for all metrics except Exclusion Error)',
             fontsize=13, fontweight='bold')

model_names = [r['Model'].replace('\n',' ') for r in results]
colors = ['#7F8C8D', '#2E86C1', '#E74C3C']

# Recall
axes[0,0].bar(model_names, [r['Recall'] for r in results], color=colors)
axes[0,0].set_title('Recall (PRIMARY)\nProportion of poor households correctly identified')
axes[0,0].set_ylim(0, 1.1)
for i, v in enumerate([r['Recall'] for r in results]):
    axes[0,0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Exclusion Error
axes[0,1].bar(model_names, [r['Exclusion Error'] for r in results], color=colors)
axes[0,1].set_title('Exclusion Error Rate (PRIMARY)\nLower is better — fewer hidden poor missed')
axes[0,1].set_ylim(0, max([r['Exclusion Error'] for r in results]) * 1.3)
for i, v in enumerate([r['Exclusion Error'] for r in results]):
    axes[0,1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

# F1
axes[1,0].bar(model_names, [r['F1-Score'] for r in results], color=colors)
axes[1,0].set_title('F1-Score')
axes[1,0].set_ylim(0, 1.1)
for i, v in enumerate([r['F1-Score'] for r in results]):
    axes[1,0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# AUC-ROC
axes[1,1].bar(model_names, [r['AUC-ROC'] for r in results], color=colors)
axes[1,1].set_title('AUC-ROC')
axes[1,1].set_ylim(0, 1.1)
for i, v in enumerate([r['AUC-ROC'] for r in results]):
    axes[1,1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/04_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 5. CONFUSION MATRICES ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')

for ax, (name, (model, X)) in zip(axes, models.items()):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Poor','Poor'],
                yticklabels=['Non-Poor','Poor'])
    ax.set_title(name.strip())
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('../outputs/figures/04_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nEvaluation complete!")
