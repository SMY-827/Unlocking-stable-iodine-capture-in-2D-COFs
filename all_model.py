import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

FIG_SIZE = (16, 12)
SUBPLOT_WSPACE = 0.3
SUBPLOT_HSPACE = 0.4

TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10

SCATTER_ALPHA = 0.6
SCATTER_SIZE = 20
REFERENCE_LINE_COLOR = 'k'
REFERENCE_LINE_STYLE = '--'
REFERENCE_LINE_WIDTH = 1.5

df = pd.read_excel('Z-score_normalization_data.xlsx', sheet_name='Sheet1')

X = df[df.columns[:-1]]
y = df['adsorption_energy']

models = {
    'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
    'XGBoost': XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
all_fold_predictions = {}

for model_name in models.keys():
    all_fold_predictions[model_name] = {
        'train': {'true': [], 'pred': []},
        'test': {'true': [], 'pred': []}
    }

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Processing the {fold+1}/5th fold...")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        r2_train = r2_score(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        mae_train = mean_absolute_error(y_train, y_train_pred)

        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)

        fold_results.append([
            fold+1, model_name,
            r2_train, rmse_train, mae_train,
            r2_test, rmse_test, mae_test
        ])

        all_fold_predictions[model_name]['train']['true'].extend(y_train.tolist())
        all_fold_predictions[model_name]['train']['pred'].extend(y_train_pred.tolist())
        all_fold_predictions[model_name]['test']['true'].extend(y_test.tolist())
        all_fold_predictions[model_name]['test']['pred'].extend(y_test_pred.tolist())

        train_df = pd.DataFrame({'True Value': y_train, 'Predicted Value': y_train_pred})
        train_df.to_csv(f'{model_name}_fold{fold+1}_train_prediction.data', sep='\t', index=False)

        test_df = pd.DataFrame({'True Value': y_test, 'Predicted Value': y_test_pred})
        test_df.to_csv(f'{model_name}_fold{fold+1}_test_prediction.data', sep='\t', index=False)

fold_df = pd.DataFrame(fold_results, columns=[
    'Fold', 'Model', 'Train_R2', 'Train_RMSE', 'Train_MAE',
    'Test_R2', 'Test_RMSE', 'Test_MAE'
])

avg_results = fold_df.groupby('Model')[['Train_R2', 'Train_RMSE', 'Train_MAE',
                                         'Test_R2', 'Test_RMSE', 'Test_MAE']].mean().reset_index()
avg_results['Fold'] = 'Average'

final_output = pd.concat([fold_df, avg_results], ignore_index=True)
final_output = final_output.sort_values(by=['Model', 'Fold'], key=lambda x: x.replace({'Average': 6}))

with open('model_performance_details.txt', 'w', encoding='utf-8') as f:
    f.write("Detailed results of five-fold cross-validation (metrics and averages for each fold)：\n\n")
    f.write(final_output.to_string(index=False, float_format='%.4f'))

print("\nAverage performance of each model：")
print(avg_results[['Model', 'Train_R2', 'Train_RMSE', 'Train_MAE',
                   'Test_R2', 'Test_RMSE', 'Test_MAE']].to_string(index=False, float_format='%.4f'))

def plot_scatter_predictions(data_dict, title, filename):
    """Plot a scatter plot of the Predicted values and True Value (all folds combined)"""
    n_models = len(data_dict)
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    axes = axes.flatten()

    for i, (model_name, data) in enumerate(data_dict.items()):
        ax = axes[i]
        y_true = np.array(data['true'])
        y_pred = np.array(data['pred'])

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        sns.scatterplot(
            x=y_true, y=y_pred,
            alpha=SCATTER_ALPHA,
            s=SCATTER_SIZE,
            color='tab:blue',
            ax=ax
        )

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val],
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            linewidth=REFERENCE_LINE_WIDTH
        )

        ax.set_title(
            f'{model_name} {title}\n(R² = {r2:.2f}, RMSE = {rmse:.2f}, MAE = {mae:.2f})',
            fontsize=TITLE_FONTSIZE,
            pad=10
        )

        ax.set_xlabel('True adsorption energy (eV)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Predicted adsorption energy (eV)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    for j in range(n_models, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

plot_scatter_predictions(
    {name: data['train'] for name, data in all_fold_predictions.items()},
    'Training set (all folds combined)',
    'COF_Eads-scatter_plot_train.png'
)

plot_scatter_predictions(
    {name: data['test'] for name, data in all_fold_predictions.items()},
    'Test set (all folds combined)',
    'COF_Eads-scatter_plot_test.png'
)

for model_name, data in all_fold_predictions.items():
    train_df = pd.DataFrame({'True Value': data['train']['true'], 'Predicted Value': data['train']['pred']})
    train_df.to_csv(f'{model_name}_allfolds_train_prediction.data', sep='\t', index=False)

    test_df = pd.DataFrame({'True Value': data['test']['true'], 'Predicted Value': data['test']['pred']})
    test_df.to_csv(f'{model_name}_allfolds_test_prediction.data', sep='\t', index=False)

print("\n=== Summary of Results ===")
print(f"1. Detailed performance metrics have been saved to 'model_performance_details.txt'")
print(f"2. Scatter plots for the training set have been saved to 'COF_Eads-scatter_plot_train.png'")
print(f"3. Scatter plots for the test set have been saved to 'COF_Eads-scatter_plot_test.png'")
print(f"4. Prediction data for each fold of each model has been saved as '_fold*.data' ")
print(f"5. Prediction data for all folds combined has been saved as '_allfolds_.data")