import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load datasets
df_red = pd.read_csv('C:\\Users\\npnit\\OneDrive\\Desktop\\C++\\winequality-red.csv', sep=';')

df_white = pd.read_csv('C:\\Users\\npnit\\OneDrive\\Desktop\\C++\\winequality-white.csv', sep=';')

# Add wine type identifiers
df_red['wine_type'] = 'red'
df_white['wine_type'] = 'white'

# ==============================================
# RED WINE ANALYSIS
# ==============================================

print("\n" + "="*50)
print("RED WINE ANALYSIS")
print("="*50 + "\n")

# EDA for red wine
plt.figure(figsize=(10,6))
sns.histplot(df_red.select_dtypes(include=[np.number]), bins=30, kde=True)
plt.title("Red Wine - Distribution of Numeric Features")
plt.show()

plt.figure(figsize=(12,8))
numeric_df_red = df_red.select_dtypes(include=np.number)
sns.heatmap(numeric_df_red.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Red Wine - Feature Correlation Heatmap")
plt.show()

# Boxplots for red wine
plt.figure(figsize=(15, 8))
numeric_cols_red = df_red.select_dtypes(include=np.number).columns.drop('quality')
df_red[numeric_cols_red].boxplot()
plt.xticks(rotation=45)
plt.title("Red Wine - Feature Distributions with Outliers")
plt.show()

# Prepare red wine data
X_red = df_red.drop(columns=['quality', 'wine_type'])
y_red = df_red['quality']

# Split red wine data
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.2, random_state=42)

# Scale red wine features
scaler_red = StandardScaler()
X_train_red = scaler_red.fit_transform(X_train_red)
X_test_red = scaler_red.transform(X_test_red)

# Hyperparameter tuning for red wine
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_red = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_red.fit(X_train_red, y_train_red)

# Best model for red wine
best_model_red = grid_search_red.best_estimator_
print("\nBest parameters for red wine:")
print(grid_search_red.best_params_)

# Predictions for red wine
y_pred_red = best_model_red.predict(X_test_red)

# Evaluation for red wine
mae_red = mean_absolute_error(y_test_red, y_pred_red)
mse_red = mean_squared_error(y_test_red, y_pred_red)
r2_red = r2_score(y_test_red, y_pred_red)

print("\nRed Wine Performance Metrics:")
print(f'Mean Absolute Error: {mae_red:.2f}')
print(f'Mean Squared Error: {mse_red:.2f}')
print(f'R² Score: {r2_red:.2f}')

# Feature importance for red wine
feature_importances_red = pd.Series(best_model_red.feature_importances_, index=X_red.columns)
feature_importances_red.nlargest(10).plot(kind='barh')
plt.title("Red Wine - Feature Importance")
plt.show()

# Save red wine model
with open("wine_quality_red_model.pkl", "wb") as file:
    pickle.dump(best_model_red, file)
print("\nRed wine model saved as wine_quality_red_model.pkl")

# ==============================================
# WHITE WINE ANALYSIS
# ==============================================

print("\n" + "="*50)
print("WHITE WINE ANALYSIS")
print("="*50 + "\n")

# EDA for white wine
plt.figure(figsize=(10,6))
sns.histplot(df_white.select_dtypes(include=[np.number]), bins=30, kde=True)
plt.title("White Wine - Distribution of Numeric Features")
plt.show()

plt.figure(figsize=(12,8))
numeric_df_white = df_white.select_dtypes(include=np.number)
sns.heatmap(numeric_df_white.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("White Wine - Feature Correlation Heatmap")
plt.show()

# Boxplots for white wine
plt.figure(figsize=(15, 8))
numeric_cols_white = df_white.select_dtypes(include=np.number).columns.drop('quality')
df_white[numeric_cols_white].boxplot()
plt.xticks(rotation=45)
plt.title("White Wine - Feature Distributions with Outliers")
plt.show()

# Prepare white wine data
X_white = df_white.drop(columns=['quality', 'wine_type'])
y_white = df_white['quality']

# Split white wine data
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white, y_white, test_size=0.2, random_state=42)

# Scale white wine features
scaler_white = StandardScaler()
X_train_white = scaler_white.fit_transform(X_train_white)
X_test_white = scaler_white.transform(X_test_white)

# Hyperparameter tuning for white wine
grid_search_white = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_white.fit(X_train_white, y_train_white)

# Best model for white wine
best_model_white = grid_search_white.best_estimator_
print("\nBest parameters for white wine:")
print(grid_search_white.best_params_)

# Predictions for white wine
y_pred_white = best_model_white.predict(X_test_white)

# Evaluation for white wine
mae_white = mean_absolute_error(y_test_white, y_pred_white)
mse_white = mean_squared_error(y_test_white, y_pred_white)
r2_white = r2_score(y_test_white, y_pred_white)

print("\nWhite Wine Performance Metrics:")
print(f'Mean Absolute Error: {mae_white:.2f}')
print(f'Mean Squared Error: {mse_white:.2f}')
print(f'R² Score: {r2_white:.2f}')

# Feature importance for white wine
feature_importances_white = pd.Series(best_model_white.feature_importances_, index=X_white.columns)
feature_importances_white.nlargest(10).plot(kind='barh')
plt.title("White Wine - Feature Importance")
plt.show()

# Save white wine model
with open("wine_quality_white_model.pkl", "wb") as file:
    pickle.dump(best_model_white, file)
print("\nWhite wine model saved as wine_quality_white_model.pkl")

print("\nAnalysis complete for both wine types!")