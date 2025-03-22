import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Load train and test data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
test_data['winddirection'] = test_data['winddirection'].bfill()

# Feature engineering for cyclical features
def create_cyclical_features(df, column, period):
    """Transform a cyclical feature into sine and cosine components."""
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / period)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / period)
    return df

# Apply cyclical transformations to 'day' and 'winddirection'
train_data = create_cyclical_features(train_data, 'day', 365)
test_data = create_cyclical_features(test_data, 'day', 365)
train_data = create_cyclical_features(train_data, 'winddirection', 360)
test_data = create_cyclical_features(test_data, 'winddirection', 360)

# Drop original cyclical columns
train_data = train_data.drop(['day', 'winddirection'], axis=1)
test_data = test_data.drop(['day', 'winddirection'], axis=1)

# Sort training data by 'id' for time-based consistency
train_data = train_data.sort_values('id')

# Separate features and target from training data
X_train_full = train_data.drop(['id', 'rainfall'], axis=1)
y_train_full = train_data['rainfall']

# Prepare test features
X_test = test_data.drop(['id'], axis=1)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_full)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
sample_weights_full = [class_weight_dict[y] for y in y_train_full]

# Define TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Define base models with class weights where applicable
rf = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
gb = GradientBoostingClassifier(random_state=42)
catboost = CatBoostClassifier(class_weights=[class_weights[0], class_weights[1]], 
                             verbose=0, random_state=42)

# Define hyperparameter grids for tuning
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5]
}
catboost_param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.1, 0.01],
    'depth': [4, 6, 8]
}

# Tune Random Forest
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
rf_grid_search.fit(X_train_full, y_train_full)
best_rf_params = rf_grid_search.best_params_
print("Best Random Forest parameters:", best_rf_params)

# Tune Gradient Boosting with sample weights
gb_grid_search = GridSearchCV(gb, gb_param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
gb_grid_search.fit(X_train_full, y_train_full, sample_weight=sample_weights_full)
best_gb_params = gb_grid_search.best_params_
print("Best Gradient Boosting parameters:", best_gb_params)

# Tune CatBoost
catboost_grid_search = GridSearchCV(catboost, catboost_param_grid, cv=tscv, 
                                   scoring='roc_auc', n_jobs=-1)
catboost_grid_search.fit(X_train_full, y_train_full)
best_catboost_params = catboost_grid_search.best_params_
print("Best CatBoost parameters:", best_catboost_params)

# Create and fit final models with best parameters
final_rf = RandomForestClassifier(**best_rf_params, class_weight=class_weight_dict, 
                                 random_state=42)
final_rf.fit(X_train_full, y_train_full)

final_gb = GradientBoostingClassifier(**best_gb_params, random_state=42)
final_gb.fit(X_train_full, y_train_full, sample_weight=sample_weights_full)

final_catboost = CatBoostClassifier(**best_catboost_params, 
                                   class_weights=[class_weights[0], class_weights[1]], 
                                   verbose=0, random_state=42)
final_catboost.fit(X_train_full, y_train_full)

# Predict probabilities on test set for each model
rf_prob = final_rf.predict_proba(X_test)[:, 1]
gb_prob = final_gb.predict_proba(X_test)[:, 1]
catboost_prob = final_catboost.predict_proba(X_test)[:, 1]

# Ensemble prediction by averaging probabilities
y_pred_prob = (rf_prob + gb_prob + catboost_prob) / 3

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': y_pred_prob
})


# Save to CSV
submission.to_csv('outputs/submission4.csv', index=False)

# Optional: Check for distribution shifts
print("\nTraining set summary statistics:\n", X_train_full.describe())
print("\nTest set summary statistics:\n", X_test.describe())
