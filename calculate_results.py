import pandas as pd

df1 = pd.read_csv('results/hopfield/exp1_results.csv')
print('--- EXP1 ---')
print(df1.groupby('corruption_level').agg({
    'success': lambda x: x.mean() * 100,
    'steps_to_converge': 'mean',
    'final_energy': 'mean'
}).round(2))

df2 = pd.read_csv('results/hopfield/exp2_results.csv')
print('\n--- EXP2 ---')
exp2_agg = df2.groupby('n_patterns')['success'].agg(['mean', 'std']).fillna(0) * 100
print(exp2_agg.round(1))
