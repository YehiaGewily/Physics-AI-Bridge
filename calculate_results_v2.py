import pandas as pd
import numpy as np

def calculate_sem(x):
    return (x.std() / np.sqrt(len(x))) * 100

print("=== EXP 1 ===")
df1 = pd.read_csv('results/hopfield/exp1_results.csv')
exp1_agg = df1.groupby('corruption_level').agg(
    strict_mean=('strict_success', lambda x: x.mean() * 100),
    strict_sem=('strict_success', calculate_sem),
    tolerant_mean=('inversion_tolerant_success', lambda x: x.mean() * 100),
    tolerant_sem=('inversion_tolerant_success', calculate_sem)
)
print(exp1_agg.round(1))

print("\n=== EXP 2 ===")
df2 = pd.read_csv('results/hopfield/exp2_results.csv')
exp2_agg = df2.groupby('n_patterns').agg(
    strict_mean=('strict_success', lambda x: x.mean() * 100),
    strict_sem=('strict_success', calculate_sem),
    tolerant_mean=('inversion_tolerant_success', lambda x: x.mean() * 100),
    tolerant_sem=('inversion_tolerant_success', calculate_sem)
).fillna(0)
print(exp2_agg.round(1))
