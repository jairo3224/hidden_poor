# ============================================================
# NOTEBOOK 01 — EXPLORATORY DATA ANALYSIS
# Run this first to understand the dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
# NOTE: Use read_csv if you have the CSV version (recommended — faster)
#       Use read_excel if you only have the XLS file from PSA
print("Loading FIES 2023 Household Summary...")
df = pd.read_csv('../data/FIES PUF 2023 Volume2 Household Summary.CSV')

# If you have XLS instead, comment the line above and uncomment this:
# df = pd.read_excel('../data/FIES PUF 2023 Volume2 Household Sum.xls')

print(f"Shape: {df.shape}") 
print(f"\nUnique region codes: {sorted(df['W_REGN'].unique())}")

# ── 2. BASIC INFO ─────────────────────────────────────────────
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== PERCAPITA STATS ===")
print(df['PERCAPITA'].describe())

print("\n=== REGION DISTRIBUTION ===")
print(df['W_REGN'].value_counts().sort_index())

print("\n=== URBAN/RURAL DISTRIBUTION ===")
print(df['URB'].value_counts())

# ── 3. BUILD POVERTY LABEL ────────────────────────────────────
# CORRECTED: Mapped exactly to actual FIES 2023 W_REGN codes
# Source: PSA 2023 Full Year Official Poverty Statistics (Aug 15, 2024)
# Cross-verified against fies_2023_vol2_metadata W_REGN value labels
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
    13: 37711,  # NCR        — National Capital Region (CORRECTED from 17)
    14: 31774,  # CAR        — Cordillera Administrative Region
    16: 29565,  # Region XIII — Caraga (CORRECTED from 13)
    17: 39150,  # MIMAROPA   — Region IV-B (CORRECTED — was missing)
    19: 30922,  # BARMM      — Bangsamoro (CORRECTED from 20)
}

df['THRESHOLD'] = df['W_REGN'].map(POVERTY_THRESHOLDS)

# Safety check — any unmapped regions?
unmapped = df['THRESHOLD'].isnull().sum()
if unmapped > 0:
    print(f"\nWARNING: {unmapped} households have unmapped region codes!")
    print("Filling with national average threshold (33,296)")
    df['THRESHOLD'] = df['THRESHOLD'].fillna(33296)

df['POOR'] = (df['PERCAPITA'] < df['THRESHOLD']).astype(int)

print("\n=== POVERTY LABEL DISTRIBUTION ===")
print(df['POOR'].value_counts())
print(f"\nOverall poverty rate: {df['POOR'].mean()*100:.2f}%")

# ── 4. VISUALIZE ──────────────────────────────────────────────
REGION_NAMES = {
    1:'I-Ilocos', 2:'II-Cagayan', 3:'III-C.Luzon',
    4:'IV-A-CALABARZON', 5:'V-Bicol', 6:'VI-W.Visayas',
    7:'VII-C.Visayas', 8:'VIII-E.Visayas', 9:'IX-Zamboanga',
    10:'X-N.Mindanao', 11:'XI-Davao', 12:'XII-SOCCSK',
    13:'NCR', 14:'CAR', 16:'XIII-Caraga',
    17:'MIMAROPA', 19:'BARMM'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('FIES 2023 — Exploratory Data Analysis', fontsize=14, fontweight='bold')

# Class distribution
counts = df['POOR'].value_counts().sort_index()
axes[0,0].bar(['Non-Poor (0)', 'Poor (1)'], counts.values, color=['#2E86C1','#E74C3C'])
axes[0,0].set_title('Poverty Label Distribution')
axes[0,0].set_ylabel('Number of Households')
for i, v in enumerate(counts.values):
    axes[0,0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# Per capita income distribution
axes[0,1].hist(df['PERCAPITA'].clip(upper=200000), bins=60,
               color='#3498DB', edgecolor='white', alpha=0.8)
axes[0,1].set_title('Per Capita Income Distribution\n(clipped at PHP 200,000)')
axes[0,1].set_xlabel('Annual Per Capita Income (PHP)')
axes[0,1].set_ylabel('Number of Households')

# Poverty rate by region
poverty_by_region = df.groupby('W_REGN')['POOR'].mean() * 100
region_labels = [REGION_NAMES.get(r, str(r)) for r in poverty_by_region.index]
bars = axes[1,0].bar(range(len(poverty_by_region)), poverty_by_region.values, color='#E74C3C')
axes[1,0].set_xticks(range(len(poverty_by_region)))
axes[1,0].set_xticklabels(region_labels, rotation=45, ha='right', fontsize=8)
axes[1,0].set_title('Poverty Rate by Region (%)')
axes[1,0].set_ylabel('Poverty Rate (%)')

# Urban vs Rural poverty
poverty_urb = df.groupby('URB')['POOR'].mean() * 100
axes[1,1].bar(['Urban (1)', 'Rural (2)'], poverty_urb.values, color=['#27AE60','#E74C3C'])
axes[1,1].set_title('Poverty Rate: Urban vs Rural')
axes[1,1].set_ylabel('Poverty Rate (%)')
for i, v in enumerate(poverty_urb.values):
    axes[1,1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nEDA complete! Figure saved to outputs/figures/01_eda_overview.png")
