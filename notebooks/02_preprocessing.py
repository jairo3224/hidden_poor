# ============================================================
# NOTEBOOK 02 — PREPROCESSING
# Builds the feature matrix and poverty label
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("Loading FIES 2023 data...")
df = pd.read_csv('../data/FIES PUF 2023 Volume2 Household Summary.CSV')



print(f"Shape: {df.shape}")

# ── 2. BUILD POVERTY LABEL ────────────────────────────────────
# CORRECTED: Mapped exactly to actual FIES 2023 W_REGN codes
# Cross-verified against dataset's actual region codes: 1-14, 16, 17, 19
POVERTY_THRESHOLDS = {
    1:  34454,  # Region I   — Ilocos Region
    2:  32160,  # Region II  — Cagayan Valley
    3:  38510,  # Region III — Central Luzon
    4:  37096,  # Region IV-A — CALABARZON
    5:  33575,  # Region V   — Bicol Region
    6:  33122,  # Region VI  — Western Visayas
    7:  34553,  # Region VII — Central Visayas
    8:  32381,  # Region VIII — Eastern Visayas
    9:  33353,  # Region IX  — Zamboanga Peninsula
    10: 33513,  # Region X   — Northern Mindanao
    11: 31521,  # Region XI  — Davao Region
    12: 29379,  # Region XII — SOCCSKSARGEN
    13: 37711,  # NCR        — National Capital Region (CORRECTED)
    14: 31774,  # CAR        — Cordillera Administrative Region
    16: 29565,  # Region XIII — Caraga (CORRECTED)
    17: 39150,  # MIMAROPA   — Region IV-B (CORRECTED — was missing)
    19: 30922,  # BARMM      — Bangsamoro (CORRECTED)
}

df['THRESHOLD'] = df['W_REGN'].map(POVERTY_THRESHOLDS)

# Safety check
unmapped = df['THRESHOLD'].isnull().sum()
if unmapped > 0:
    print(f"WARNING: {unmapped} unmapped region codes — filling with national average")
    df['THRESHOLD'] = df['THRESHOLD'].fillna(33296)

df['POOR'] = (df['PERCAPITA'] < df['THRESHOLD']).astype(int)
print(f"\nPoverty label built:")
print(f"  Poor (1):     {df['POOR'].sum():,} households ({df['POOR'].mean()*100:.1f}%)")
print(f"  Non-Poor (0): {(df['POOR']==0).sum():,} households ({(1-df['POOR'].mean())*100:.1f}%)")

# ── 3. SELECT FEATURES ───────────────────────────────────────
# EXCLUDED: all income variables — target leakage
# EXCLUDED: TOTEX, FOOD, NFOOD — expenditure aggregates too close to income
# EXCLUDED: IMPUTED_RENT, BIMPUTED_RENT — derived/imputed values
# EXCLUDED: PERCAPITA, NPCINC, etc. — used for label construction only
# INCLUDED: specific expenditure categories + demographics

FEATURE_COLS = [
    # Demographics
    'FSIZE',         # Average family size
    'URB',           # Urban (1) / Rural (2)
    'W_REGN',        # Region

    # Food expenditure (specific — NOT the FOOD aggregate total)
    'BREAD',         # Bread and cereals
    'MEAT',          # Meat
    'FISH',          # Fish and seafood
    'MILK',          # Milk, cheese, eggs
    'OIL',           # Oils and fats
    'FRUIT',         # Fruit
    'VEG',           # Vegetables
    'SUGAR',         # Sugar, jam, honey, etc.
    'FOOD_NEC',      # Food products NEC

    # Non-food expenditure
    'CLOTH',         # Clothing and footwear
    'HEALTH',        # Health expenditure
    'TRANSPORT',     # Transport expenditure
    'COMMUNICATION', # Communication expenditure
    'RECREATION',    # Recreation and culture
    'EDUCATION',     # Education expenditure
    'INSURANCE',     # Insurance
    'DURABLE',       # Durable furniture and equipment
    'FURNISHING',    # Furnishings and maintenance
    'ACTRENT',       # Actual house rent paid
    'ALCOHOL',       # Alcoholic beverages
    'TOBACCO',       # Tobacco
    'MISCELLANEOUS', # Miscellaneous goods and services
]

# Verify all columns exist in the dataset
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    print(f"\nWARNING: These columns not found: {missing_cols}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

print(f"\nFeature columns selected: {len(FEATURE_COLS)}")

# ── 4. BUILD FEATURE MATRIX ──────────────────────────────────
X = df[FEATURE_COLS].copy()
y = df['POOR'].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"\nMissing values per column:")
missing = X.isnull().sum()
print(missing[missing > 0] if missing.any() else "  None — all features complete!")

# ── 5. HANDLE MISSING VALUES ─────────────────────────────────
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())
        print(f"  Filled {col} with median")

# ── 6. CLASS IMBALANCE CHECK ─────────────────────────────────
poor_n    = y.sum()
nonpoor_n = (y == 0).sum()
ratio     = nonpoor_n / poor_n
print(f"\nClass imbalance ratio: {ratio:.2f}:1 (non-poor to poor)")
if ratio > 3:
    print("  → Will use class_weight='balanced' in all models")
    print("  → NOT using SMOTE (generates synthetic households — bad for survey data)")

# ── 7. SAVE PROCESSED DATA ───────────────────────────────────
X.to_csv('../data/X_features.csv', index=False)
y.to_csv('../data/y_labels.csv', index=False)
print(f"\nSaved X_features.csv ({X.shape[0]:,} rows x {X.shape[1]} features)")
print(f"Saved y_labels.csv ({y.sum():,} poor, {(y==0).sum():,} non-poor)")
print("\nPreprocessing complete!")
