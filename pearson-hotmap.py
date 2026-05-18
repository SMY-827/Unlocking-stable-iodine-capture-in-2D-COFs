import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
excel_file = pd.ExcelFile('Z-score_normalization_data.xlsx')
df = excel_file.parse('Sheet1')
correlation_matrix = df[df.columns[:]].corr()
mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', mask=~mask)
plt.title('correlation_heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
correlation_results = {}
for feature in df.columns[:-1]:
    corr_coef, p_value = pearsonr(df[feature], df['adsorption_energy'])
    correlation_results[feature] = {'Correlation Coefficient': corr_coef, 'P-value': p_value}

correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index')
with open('pearson_results.txt', 'w') as file:
    file.write(correlation_df.to_csv(sep='\t', na_rep='nan'))
