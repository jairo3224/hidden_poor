# ============================================================
# NOTEBOOK 03 — MODEL TRAINING
# Trains Logistic Regression, CatBoost, XGBoost
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading preprocessed data...")
X = pd.read_csv('../data/X_features.csv')
y = pd.read_csv('../data/y_labels.csv').squeeze()

print(f"X shape: {X.shape}")
print(f"y distribution:\n{y.value_counts()}")

# ── 1. TRAIN/TEST SPLIT ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y  # preserve class ratio
)
print(f"\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── 2. IDENTIFY CATEGORICAL FEATURES ─────────────────────────
CAT_FEATURES = ['URB', 'W_REGN']
cat_indices = [X.columns.tolist().index(c) for c in CAT_FEATURES if c in X.columns]

# ── 3. SCALE FOR LOGISTIC REGRESSION ────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, '../outputs/models/scaler.pkl')

# ── 4. STRATIFIED 5-FOLD CV SETUP ────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 5. LOGISTIC REGRESSION (Baseline) ────────────────────────
print("\n--- Training Logistic Regression (Baseline) ---")
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # handles class imbalance without SMOTE
    random_state=42,
    n_jobs=-1
)
lr_cv = cross_val_score(lr, X_train_scaled, y_train, cv=skf, scoring='recall', n_jobs=-1)
print(f"LR CV Recall: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, '../outputs/models/logistic_regression.pkl')
print("Logistic Regression saved!")

# ── 6. CATBOOST ──────────────────────────────────────────────
print("\n--- Training CatBoost ---")
cb = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='Recall',
    random_seed=42,
    verbose=100,
    class_weights={0: 1, 1: int((y_train==0).sum() / y_train.sum())},
    cat_features=cat_indices,
    early_stopping_rounds=50
)
# CatBoost needs a validation set for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
)
cb.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    use_best_model=True
)
cb_cv = cross_val_score(
    CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6,
                       random_seed=42, verbose=0),
    X_train, y_train, cv=skf, scoring='recall', n_jobs=-1,
    params={'cat_features': cat_indices}
)
print(f"CatBoost CV Recall: {cb_cv.mean():.4f} ± {cb_cv.std():.4f}")
cb.save_model('../outputs/models/catboost_model.cbm')
print("CatBoost saved!")

# ── 7. XGBOOST ───────────────────────────────────────────────
print("\n--- Training XGBoost ---")
scale_pos_weight = int((y_train==0).sum() / y_train.sum())
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,  # handles imbalance
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50,
    verbosity=1,
    n_jobs=-1
)
xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=100
)
xgb_cv = cross_val_score(
    XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                  scale_pos_weight=scale_pos_weight, random_state=42,
                  verbosity=0, n_jobs=-1),
    X_train, y_train, cv=skf, scoring='recall', n_jobs=-1
)
print(f"XGBoost CV Recall: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
joblib.dump(xgb, '../outputs/models/xgboost_model.pkl')
print("XGBoost saved!")

# ── 8. SAVE TEST SET ─────────────────────────────────────────
X_test.to_csv('../data/X_test.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)
print("\nTest set saved for evaluation.")
print("\nAll models trained successfully!")
