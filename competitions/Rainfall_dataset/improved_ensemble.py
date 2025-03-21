import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

# 1. Advanced ML Model with Ensemble including Random Forest
def preprocess_data(df):
    # Fill NA with mean
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Handle 'day' - extract cyclical features
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    
    return df.drop('day', axis=1)

def train_ensemble_model():
    # Read and preprocess data
    df = pd.read_csv('data/train.csv')
    df = preprocess_data(df)
    
    # Features and target
    X = df.drop(['rainfall', 'id'], axis=1)
    y = df['rainfall']
    
    # Handle unbalanced data with SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, 
                                                        test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grids for tuning
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    
    lgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'verbose': [-1]
    }
    
    cat_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Initialize models
    xgb_model = xgb.XGBClassifier(random_state=42)
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    cat_model = CatBoostClassifier(random_state=42, verbose=0)
    rf_model = RandomForestClassifier(random_state=42)
    
    # Perform parameter tuning
    xgb_search = RandomizedSearchCV(xgb_model, xgb_params, n_iter=10, cv=3, random_state=42)
    lgb_search = RandomizedSearchCV(lgb_model, lgb_params, n_iter=10, cv=3, random_state=42, verbose=-1)
    cat_search = RandomizedSearchCV(cat_model, cat_params, n_iter=10, cv=3, random_state=42)
    rf_search = RandomizedSearchCV(rf_model, rf_params, n_iter=10, cv=3, random_state=42)
    
    # Fit models
    xgb_search.fit(X_train_scaled, y_train)
    lgb_search.fit(X_train_scaled, y_train)
    cat_search.fit(X_train_scaled, y_train)
    rf_search.fit(X_train_scaled, y_train)
    
    # Get best models
    best_xgb = xgb_search.best_estimator_
    best_lgb = lgb_search.best_estimator_
    best_cat = cat_search.best_estimator_
    best_rf = rf_search.best_estimator_
    
    # Create ensemble with top 3 performing models
    estimators = [
        ('xgb', best_xgb),
        ('lgb', best_lgb),
        ('cat', best_cat),
        ('rf', best_rf)
    ]
    
    ensemble_model = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )
    
    # Train ensemble
    ensemble_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = ensemble_model.predict(X_test_scaled)
    y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
    
    print("Ensemble Model Performance:")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    print(f"Best LightGBM params: {lgb_search.best_params_}")
    print(f"Best CatBoost params: {cat_search.best_params_}")
    print(f"Best RandomForest params: {rf_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return ensemble_model, scaler

def predict_with_ensemble(model, scaler, test_file, output_file):
    test_df = pd.read_csv(test_file)
    test_ids = test_df['id']
    test_df = preprocess_data(test_df)
    X_test = test_df.drop('id', axis=1)
    
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict_proba(X_test_scaled)[:, 1]
    
    output_df = pd.DataFrame({
        'id': test_ids,
        'rainfall': predictions
    })
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# 2. Deep Neural Network Approach with preprocessing
class WeatherDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        if y is not None:
            self.y = torch.FloatTensor(y)
        else:
            self.y = None
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class WeatherNN(nn.Module):
    def __init__(self, input_size):
        super(WeatherNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_dnn_model():
    df = pd.read_csv('data/train.csv')
    df = preprocess_data(df)
    
    X = df.drop(['rainfall', 'id'], axis=1)
    y = df['rainfall'].values
    
    # Handle unbalanced data
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, 
                                                        test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets and loaders
    train_dataset = WeatherDataset(X_train_scaled, y_train)
    test_dataset = WeatherDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = WeatherNN(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(outputs.squeeze().cpu().numpy())
        
        auc = roc_auc_score(y_true, y_pred)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, AUC: {auc:.4f}")
    
    return model, scaler

def predict_with_dnn(model, scaler, test_file, output_file):
    test_df = pd.read_csv(test_file)
    test_ids = test_df['id']
    test_df = preprocess_data(test_df)
    X_test = test_df.drop('id', axis=1)
    
    X_test_scaled = scaler.transform(X_test)
    test_dataset = WeatherDataset(X_test_scaled)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    
    with torch.no_grad():
        for batch_X in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().cpu().numpy())
    
    output_df = pd.DataFrame({
        'id': test_ids,
        'rainfall': predictions
    })
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # 1. Ensemble Model with Random Forest
    print("Training Ensemble Model...")
    ensemble_model, ensemble_scaler = train_ensemble_model()
    predict_with_ensemble(ensemble_model, ensemble_scaler, 'data/test.csv', 'ensemble_predictions.csv')
    
    # 2. DNN Model
    print("\nTraining Deep Neural Network...")
    dnn_model, dnn_scaler = train_dnn_model()
    predict_with_dnn(dnn_model, dnn_scaler, 'data/test.csv', 'dnn_predictions.csv')
