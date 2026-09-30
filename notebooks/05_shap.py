# ============================================================
# NOTEBOOK 05 — SHAP EXPLAINABILITY ANALYSIS
# Global feature importance + Hidden Poor profiling
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA AND BEST MODEL ──────────────────────────────
print("Loading data and models...")
X_test = pd.read_csv('../data/X_test.csv')
y_test = pd.read_csv('../data/y_test.csv').squeeze()

# Load both models — we'll run SHAP on CatBoost (primary) and XGBoost
cb  = CatBoostClassifier(); cb.load_model('../outputs/models/catboost_model.cbm')
xgb = joblib.load('../outputs/models/xgboost_model.pkl')

# Human-readable feature names
FEATURE_NAMES = {
    'FSIZE':         'Household Size',
    'URB':           'Urban/Rural',
    'W_REGN':        'Region',
    'BREAD':         'Bread & Cereals Expenditure',
    'MEAT':          'Meat Expenditure',
    'FISH':          'Fish & Seafood Expenditure',
    'MILK':          'Dairy Expenditure',
    'OIL':           'Oils & Fats Expenditure',
    'FRUIT':         'Fruit Expenditure',
    'VEG':           'Vegetable Expenditure',
    'SUGAR':         'Sugar & Sweets Expenditure',
    'FOOD_NEC':      'Other Food Expenditure',
    'CLOTH':         'Clothing Expenditure',
    'HEALTH':        'Health Expenditure',
    'TRANSPORT':     'Transport Expenditure',
    'COMMUNICATION': 'Communication Expenditure',
    'RECREATION':    'Recreation Expenditure',
    'EDUCATION':     'Education Expenditure',
    'INSURANCE':     'Insurance Expenditure',
    'DURABLE':       'Durable Goods Expenditure',
    'FURNISHING':    'Furnishing Expenditure',
    'ACTRENT':       'Actual Rent Paid',
    'ALCOHOL':       'Alcohol Expenditure',
    'TOBACCO':       'Tobacco Expenditure',
    'MISCELLANEOUS': 'Miscellaneous Expenditure',
}
X_test_named = X_test.rename(columns=FEATURE_NAMES)

# ── 2. CATBOOST SHAP — GLOBAL ─────────────────────────────────
print("\nComputing CatBoost SHAP values (global)...")
cb_explainer = shap.TreeExplainer(cb)
cb_shap = cb_explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(cb_shap, X_test_named, show=False, max_display=20)
plt.title('CatBoost — Global Feature Importance (SHAP)\nImpact on Poverty Classification', fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/05_catboost_shap_global.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_catboost_shap_global.png")

# ── 3. XGBOOST SHAP — GLOBAL ──────────────────────────────────
print("\nComputing XGBoost SHAP values (global)...")
xgb_explainer = shap.TreeExplainer(xgb)
xgb_shap = xgb_explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(xgb_shap, X_test_named, show=False, max_display=20)
plt.title('XGBoost — Global Feature Importance (SHAP)\nImpact on Poverty Classification', fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/05_xgboost_shap_global.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_xgboost_shap_global.png")

# ── 4. HIDDEN POOR ANALYSIS ───────────────────────────────────
# Hidden poor = actually poor BUT model predicted non-poor (false negatives)
print("\nIdentifying the Hidden Poor...")

cb_preds = cb.predict(X_test)

# Create analysis dataframe
analysis_df = X_test.copy()
analysis_df['actual_poor']     = y_test.values
analysis_df['cb_predicted']    = cb_preds
analysis_df['hidden_poor']     = ((analysis_df['actual_poor'] == 1) &
                                   (analysis_df['cb_predicted'] == 0)).astype(int)

n_hidden = analysis_df['hidden_poor'].sum()
n_actual_poor = analysis_df['actual_poor'].sum()
print(f"\nHidden Poor (CatBoost): {n_hidden:,} out of {n_actual_poor:,} actual poor households")
print(f"Exclusion Error Rate: {n_hidden/n_actual_poor*100:.2f}%")

# SHAP for hidden poor vs correctly identified poor
hidden_idx  = analysis_df[analysis_df['hidden_poor'] == 1].index
correct_idx = analysis_df[(analysis_df['actual_poor'] == 1) &
                           (analysis_df['cb_predicted'] == 1)].index

print(f"\nComparing SHAP profiles:")
print(f"  Hidden poor (misclassified): {len(hidden_idx)}")
print(f"  Correctly identified poor:   {len(correct_idx)}")

if len(hidden_idx) > 10:
    # SHAP bar plot for hidden poor
    hidden_X = X_test.loc[hidden_idx]
    hidden_shap = cb_explainer.shap_values(hidden_X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(hidden_shap, hidden_X.rename(columns=FEATURE_NAMES),
                      plot_type='bar', show=False, max_display=15)
    plt.title(f'Hidden Poor Profile — CatBoost SHAP\n({len(hidden_idx)} misclassified poor households)',
              fontweight='bold')
    plt.tight_layout()
    plt.savefig('../outputs/figures/05_hidden_poor_shap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: 05_hidden_poor_shap.png")

# ── 5. EXPENDITURE PROFILE COMPARISON ────────────────────────
expenditure_cols = ['BREAD','MEAT','FISH','MILK','OIL','FRUIT','VEG',
                    'SUGAR','CLOTH','HEALTH','TRANSPORT','COMMUNICATION',
                    'RECREATION','EDUCATION','INSURANCE','DURABLE',
                    'FURNISHING','ACTRENT','ALCOHOL','TOBACCO']

expenditure_cols = [c for c in expenditure_cols if c in X_test.columns]

hidden_mean   = analysis_df.loc[hidden_idx, expenditure_cols].mean()
correct_mean  = analysis_df.loc[correct_idx, expenditure_cols].mean()

compare_df = pd.DataFrame({
    'Hidden Poor (Missed)':     hidden_mean,
    'Correctly Identified Poor': correct_mean,
}).rename(index=FEATURE_NAMES)

fig, ax = plt.subplots(figsize=(12, 7))
compare_df.plot(kind='bar', ax=ax, color=['#E74C3C','#2E86C1'])
ax.set_title('Average Expenditure Profile\nHidden Poor vs Correctly Identified Poor',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Expenditure Category')
ax.set_ylabel('Average Annual Expenditure (PHP)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/figures/05_expenditure_profile_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_expenditure_profile_comparison.png")

# ── 6. SAVE HIDDEN POOR DATA ─────────────────────────────────
hidden_df = analysis_df[analysis_df['hidden_poor'] == 1]
hidden_df.to_csv('../outputs/results/hidden_poor_records.csv', index=False)
print(f"\nSaved {len(hidden_df)} hidden poor records to results/hidden_poor_records.csv")
print("\nSHAP analysis complete!")
