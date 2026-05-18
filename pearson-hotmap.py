import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 读取文件
excel_file = pd.ExcelFile('Z-score_normalization_data.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')

# 计算相关系数矩阵
correlation_matrix = df[df.columns[:]].corr()

# 创建下三角矩阵的掩码
mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

# 绘制只显示下三角（包含对角线）的热图
plt.figure(figsize=(16, 14))
# 设置 fmt 参数为 '.2f' 以保留两位小数
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', mask=~mask)
plt.title('correlation_heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# 计算 pearson 系数和 P 值并保存到 txt 文件
correlation_results = {}
for feature in df.columns[:-1]:
    corr_coef, p_value = pearsonr(df[feature], df['adsorption_energy'])
    correlation_results[feature] = {'Correlation Coefficient': corr_coef, 'P-value': p_value}

correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index')
with open('pearson_results.txt', 'w') as file:
    file.write(correlation_df.to_csv(sep='\t', na_rep='nan'))