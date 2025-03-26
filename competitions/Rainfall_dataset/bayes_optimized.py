import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

# Data loading
train = pd.read_csv("data/rainfall_transformed.csv")  # Training data
val = pd.read_csv("data/train.csv")                  # Validation data
test = pd.read_csv("data/test.csv")                  # Test data

# ======= 1. Improved Feature Engineering =======
# Create new variables
for df in [train, val, test]:
    # Change rates of original variables
    df['temp_change_rate'] = df['temparature'].diff().fillna(0)
    df['humidity_change_trend'] = df['humidity'].diff().fillna(0)
    df['pressure_change'] = df['pressure'].diff().fillna(0)
    df['windspeed_change'] = df['windspeed'].diff().fillna(0)
    
    # Temperature range
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    
    # Humidity to temperature ratio
    df['humidity_temp_ratio'] = df['humidity'] / (df['temparature'] + 0.1)  # avoid division by zero
    
    # Wind and pressure interaction
    df['wind_pressure_interaction'] = df['windspeed'] * df['pressure'] / 1000
    
    # Cloud and sunshine interaction
    df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 0.1)  # avoid division by zero
    
    # Dewpoint and temperature difference
    df['dewpoint_temp_diff'] = df['temparature'] - df['dewpoint']
    
    # Add lag features (for important variables)
    for lag in [1, 2]:
        df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag).fillna(0)
        df[f'pressure_lag_{lag}'] = df['pressure'].shift(lag).fillna(0)
        df[f'temp_lag_{lag}'] = df['temparature'].shift(lag).fillna(0)

# Fill NA values with 0
train = train.fillna(0)
val = val.fillna(0)
test = test.fillna(0)

RMV = ['rainfall', 'id']
FEATURES = [c for c in train.columns if c not in RMV]

# ======= 2. Bayesian Optimization for XGBoost =======
def xgb_cv(max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, min_child_weight):
    """Bayesian optimization objective function"""
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'min_child_weight': min_child_weight,
        'n_estimators': 10000,
        'eval_metric': 'auc',
        'early_stopping_rounds': 100
    }
    
    cv_scores = []
    train_auc_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(train):
        x_train = train.loc[train_idx, FEATURES]
        y_train = train.loc[train_idx, 'rainfall']
        x_val = train.loc[val_idx, FEATURES]
        y_val = train.loc[val_idx, 'rainfall']
        
        model = XGBClassifier(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            verbose=False
        )
        
        train_pred = model.predict_proba(x_train)[:, 1]
        val_pred = model.predict_proba(x_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        
        cv_scores.append(val_auc)
        train_auc_scores.append(train_auc)
    
    mean_train_auc = np.mean(train_auc_scores)
    mean_val_auc = np.mean(cv_scores)
    overfit_gap = mean_train_auc - mean_val_auc
    print(f"Train AUC = {mean_train_auc:.3f}, Val AUC = {mean_val_auc:.3f}, Overfitting = {overfit_gap:.3f}")
    
    if overfit_gap > 0.05:
        return mean_val_auc * 0.95  # Penalize overfitting
    elif overfit_gap > 0.1:
        return mean_val_auc * 0.9
    
    return mean_val_auc

# ======= 3. Bayesian Optimization for Random Forest =======
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    """Bayesian optimization objective function for Random Forest"""
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'random_state': 42,
        'n_jobs': -1
    }
    
    cv_scores = []
    train_auc_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(train):
        x_train = train.loc[train_idx, FEATURES]
        y_train = train.loc[train_idx, 'rainfall']
        x_val = train.loc[val_idx, FEATURES]
        y_val = train.loc[val_idx, 'rainfall']
        
        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)
        
        train_pred = model.predict_proba(x_train)[:, 1]
        val_pred = model.predict_proba(x_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        
        cv_scores.append(val_auc)
        train_auc_scores.append(train_auc)
    
    mean_train_auc = np.mean(train_auc_scores)
    mean_val_auc = np.mean(cv_scores)
    overfit_gap = mean_train_auc - mean_val_auc
    print(f"RF Train AUC = {mean_train_auc:.3f}, Val AUC = {mean_val_auc:.3f}, Overfitting = {overfit_gap:.3f}")
    
    return mean_val_auc

# ======= 4. Run Bayesian Optimization for XGBoost =======
xgb_pbounds = {
    'max_depth': (3, 12),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.7, 1.0),
    'colsample_bytree': (0.7, 1.0),
    'gamma': (0, 2),
    'reg_alpha': (0.01, 5),
    'reg_lambda': (0.01, 5),
    'min_child_weight': (1, 10)
}

print("\n=== Starting XGBoost Optimization ===")
xgb_optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=xgb_pbounds,
    random_state=42,
    verbose=2
)

xgb_optimizer.maximize(init_points=15, n_iter=40)

# Get best XGBoost parameters
xgb_best_params = xgb_optimizer.max['params']
xgb_best_params['max_depth'] = int(xgb_best_params['max_depth'])
print('Best XGBoost parameters:', xgb_best_params)

# ======= 5. Run Bayesian Optimization for Random Forest =======
rf_pbounds = {
    'n_estimators': (50, 300),
    'max_depth': (3, 15),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

print("\n=== Starting Random Forest Optimization ===")
rf_optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=rf_pbounds,
    random_state=42,
    verbose=2
)

rf_optimizer.maximize(init_points=15, n_iter=40)

# Get best Random Forest parameters
rf_best_params = rf_optimizer.max['params']
rf_best_params['n_estimators'] = int(rf_best_params['n_estimators'])
rf_best_params['max_depth'] = int(rf_best_params['max_depth'])
print('Best Random Forest parameters:', rf_best_params)

# ======= 6. Cross-validation with optimized models =======
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Prepare arrays for predictions
oof_xgb = np.zeros(len(val))  # Validation set predictions
pred_xgb = np.zeros(len(test))
oof_rf = np.zeros(len(val))
pred_rf = np.zeros(len(test))

xgb_fold_train_auc = []
xgb_fold_val_auc = []
rf_fold_train_auc = []
rf_fold_val_auc = []

for i, (train_index, val_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index, FEATURES]
    y_train = train.loc[train_index, "rainfall"]
    x_valid = val.loc[val_index % len(val), FEATURES]  # Adjust indices for validation set
    y_valid = val.loc[val_index % len(val), "rainfall"]
    x_test = test[FEATURES]

    # XGBoost model
    xgb_model = XGBClassifier(
        max_depth=int(xgb_best_params['max_depth']),
        learning_rate=xgb_best_params['learning_rate'],
        subsample=xgb_best_params['subsample'],
        colsample_bytree=xgb_best_params['colsample_bytree'],
        gamma=xgb_best_params['gamma'],
        reg_alpha=xgb_best_params['reg_alpha'],
        reg_lambda=xgb_best_params['reg_lambda'],
        min_child_weight=xgb_best_params['min_child_weight'],
        n_estimators=10000,
        early_stopping_rounds=100,
        eval_metric='auc'
    )
    
    xgb_model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )
    
    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=int(rf_best_params['n_estimators']),
        max_depth=int(rf_best_params['max_depth']),
        min_samples_split=int(rf_best_params['min_samples_split']),
        min_samples_leaf=int(rf_best_params['min_samples_leaf']),
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(x_train, y_train)
    
    # XGBoost predictions
    xgb_train_pred = xgb_model.predict_proba(x_train)[:, 1]
    xgb_val_pred = xgb_model.predict_proba(x_valid)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(x_test)[:, 1]
    
    # Random Forest predictions
    rf_train_pred = rf_model.predict_proba(x_train)[:, 1]
    rf_val_pred = rf_model.predict_proba(x_valid)[:, 1]
    rf_test_pred = rf_model.predict_proba(x_test)[:, 1]
    
    # Store XGBoost metrics
    xgb_train_auc = roc_auc_score(y_train, xgb_train_pred)
    xgb_val_auc = roc_auc_score(y_valid, xgb_val_pred)
    xgb_fold_train_auc.append(xgb_train_auc)
    xgb_fold_val_auc.append(xgb_val_auc)
    
    # Store Random Forest metrics
    rf_train_auc = roc_auc_score(y_train, rf_train_pred)
    rf_val_auc = roc_auc_score(y_valid, rf_val_pred)
    rf_fold_train_auc.append(rf_train_auc)
    rf_fold_val_auc.append(rf_val_auc)
    
    # Store out-of-fold predictions (for validation set)
    oof_xgb[val_index % len(val)] = xgb_val_pred
    oof_rf[val_index % len(val)] = rf_val_pred
    
    # Accumulate test predictions
    pred_xgb += xgb_test_pred / FOLDS
    pred_rf += rf_test_pred / FOLDS
    
    print(f"XGBoost - Train AUC: {xgb_train_auc:.4f}, Val AUC: {xgb_val_auc:.4f}")
    print(f"Random Forest - Train AUC: {rf_train_auc:.4f}, Val AUC: {rf_val_auc:.4f}")

# ======= 7. Optimize ensemble weights =======
def optimize_weights(oof_preds, y_true):
    """Find optimal weights for model blending using simple grid search"""
    best_score = 0
    best_weights = None
    
    for w1 in range(0, 11, 1):  # 0 to 1 with 0.1 step
        w1 = w1 / 10
        w2 = 1 - w1
        
        blend_preds = w1 * oof_preds[0] + w2 * oof_preds[1]
        score = roc_auc_score(y_true, blend_preds)
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2)
    
    return best_weights, best_score

# Optimize weights for XGBoost and RF ensemble using validation set
weights, score = optimize_weights([oof_xgb, oof_rf], val['rainfall'])
print(f"Optimized ensemble weights: XGBoost={weights[0]:.2f}, RF={weights[1]:.2f}, Score={score:.4f}")

# ======= 8. Dynamic threshold optimization =======
def find_optimal_thresholds(oof_preds, y_true):
    """Find upper and lower thresholds for classification adjustment"""
    best_score = roc_auc_score(y_true, oof_preds)
    best_thresholds = (0.05, 0.95)  # Default thresholds
    
    for lower in range(1, 15, 1):  # 0.01 to 0.15
        lower = lower / 100
        for upper in range(85, 100, 1):  # 0.85 to 0.99
            upper = upper / 100
            
            adjusted_preds = oof_preds.copy()
            adjusted_preds[adjusted_preds < lower] = 0
            adjusted_preds[adjusted_preds > upper] = 1
            
            score = roc_auc_score(y_true, adjusted_preds)
            
            if score > best_score:
                best_score = score
                best_thresholds = (lower, upper)
    
    return best_thresholds, best_score

# Calculate ensemble OOF predictions for validation set
oof_ensemble = weights[0] * oof_xgb + weights[1] * oof_rf

# Find optimal thresholds
thresholds, threshold_score = find_optimal_thresholds(oof_ensemble, val['rainfall'])
print(f"Optimal thresholds: Lower={thresholds[0]:.2f}, Upper={thresholds[1]:.2f}, Score={threshold_score:.4f}")

# ======= 9. Create final predictions and submission file =======
# Combine XGBoost and Random Forest predictions with optimal weights
final_pred = weights[0] * pred_xgb + weights[1] * pred_rf

# Try to load external best submission if available (for ensemble)
try:
    best_public = pd.read_csv("/kaggle/input/0-96245-lda-lgs-ensemble-for-rainfall-pred/submission.csv")
    best_public = best_public.rainfall.values
    
    # Enhanced ensemble with external prediction
    ensemble_pred = 0.7 * rankdata(final_pred) + 0.3 * rankdata(best_public)
    ensemble_pred = rankdata(ensemble_pred) / len(ensemble_pred)
    
    print("Successfully integrated external predictions in ensemble")
    final_pred = ensemble_pred
except:
    print("External predictions not available, using only current models")

# Apply optimized thresholds to final predictions
final_pred_adjusted = final_pred.copy()
final_pred_adjusted[final_pred < thresholds[0]] = 0
final_pred_adjusted[final_pred > thresholds[1]] = 1

# Create submission file
sample = pd.read_csv("outputs/sample_submission.csv")
sample.rainfall = final_pred_adjusted
sample.to_csv("outputs/submission_26.03.csv", index=False)
print("Final submission file created")

# ======= 10. Model performance analysis =======
print("\n===== Model Performance Analysis =====")
xgb_mean_train_auc = np.mean(xgb_fold_train_auc)
xgb_mean_val_auc = np.mean(xgb_fold_val_auc)
xgb_overfit_gap = xgb_mean_train_auc - xgb_mean_val_auc

rf_mean_train_auc = np.mean(rf_fold_train_auc)
rf_mean_val_auc = np.mean(rf_fold_val_auc)
rf_overfit_gap = rf_mean_train_auc - rf_mean_val_auc

print(f"XGBoost - Train AUC: {xgb_mean_train_auc:.4f}, Val AUC: {xgb_mean_val_auc:.4f}, Gap: {xgb_overfit_gap:.4f}")
print(f"Random Forest - Train AUC: {rf_mean_train_auc:.4f}, Val AUC: {rf_mean_val_auc:.4f}, Gap: {rf_overfit_gap:.4f}")
print(f"Ensemble - Val AUC: {score:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
models = ['XGBoost', 'Random Forest']
train_aucs = [xgb_mean_train_auc, rf_mean_train_auc]
val_aucs = [xgb_mean_val_auc, rf_mean_val_auc]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, train_aucs, width, label='Train AUC')
plt.bar(x + width/2, val_aucs, width, label='Validation AUC')

plt.xlabel('Models')
plt.ylabel('AUC Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models)
plt.ylim(0.5, 1.0)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
