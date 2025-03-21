import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTEENN
import pickle
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, is_training=True, reference_data=None):
    """
    Process the dataframe by handling missing values and creating features
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input data
    is_training : bool
        Whether this is training data (True) or test data (False)
    reference_data : dict, optional
        Dictionary containing medians and other statistics from training data
        
    Returns:
    --------
    processed_df : pandas DataFrame
        The processed dataframe
    stats : dict
        Statistics from the data (only for training data)
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Store statistics if in training mode
    stats = {}
    
    # Handle missing values
    if is_training:
        # In training mode, calculate and store medians
        stats['medians'] = processed_df.median()
        processed_df = processed_df.fillna(stats['medians'])
    else:
        # In test mode, use medians from training data
        processed_df = processed_df.fillna(reference_data['medians'])
    
    # ---- FEATURE ENGINEERING ----
    
    # 1. Convert day of year to cyclical month feature
    processed_df['month'] = processed_df['day'].apply(lambda x: datetime.datetime.strptime(str(x), '%j').month)
    processed_df['month_sin'] = np.sin(2 * np.pi * processed_df['month'] / 12)
    processed_df['month_cos'] = np.cos(2 * np.pi * processed_df['month'] / 12)
    
    # 2. Add season indicator (meteorological seasons)
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    processed_df['season'] = processed_df['month'].apply(get_season)
    processed_df = pd.get_dummies(processed_df, columns=['season'], prefix='season')
    
    # Ensure all season columns exist
    for col in ['season_0', 'season_1', 'season_2', 'season_3']:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    # 3. Temperature-related features
    processed_df['temp_range'] = processed_df['maxtemp'] - processed_df['mintemp']
    processed_df['temp_deviation'] = processed_df['temparature'] - ((processed_df['maxtemp'] + processed_df['mintemp']) / 2)
    
    # 4. Humidity and pressure interactions (avoid division by zero)
    processed_df['humidity_pressure_ratio'] = processed_df['humidity'] / processed_df['pressure'].replace(0, 0.001)
    processed_df['dewpoint_diff'] = processed_df['temparature'] - processed_df['dewpoint']
    
    # 5. Wind features
    processed_df['is_windy'] = (processed_df['windspeed'] > 30).astype(int)
    processed_df['wind_chill'] = 13.12 + 0.6215*processed_df['temparature'] - 11.37*(processed_df['windspeed']**0.16 + 0.001) + 0.3965*processed_df['temparature']*(processed_df['windspeed']**0.16 + 0.001)
    processed_df['wind_direction_rad'] = np.radians(processed_df['winddirection'])
    processed_df['wind_x'] = processed_df['windspeed'] * np.cos(processed_df['wind_direction_rad'])
    processed_df['wind_y'] = processed_df['windspeed'] * np.sin(processed_df['wind_direction_rad'])
    
    # 6. Create cloud-humidity interaction
    processed_df['cloud_humidity_product'] = processed_df['cloud'] * processed_df['humidity'] / 100
    
    # Sort by day for proper time-series handling
    processed_df = processed_df.sort_values('day')
    
    # 7. Add time series features - PROPERLY HANDLING TEMPORAL EFFECTS
    # We use rolling windows instead of simple shifts to avoid look-ahead bias
    for col in ['pressure', 'humidity', 'cloud', 'temparature', 'windspeed']:
        # Rolling mean and std over previous 3, 7, and 14 days (avoiding future data)
        processed_df[f'{col}_roll_mean_3'] = processed_df[col].rolling(window=3, min_periods=1).mean().shift(1)
        processed_df[f'{col}_roll_mean_7'] = processed_df[col].rolling(window=7, min_periods=1).mean().shift(1)
        processed_df[f'{col}_roll_std_7'] = processed_df[col].rolling(window=7, min_periods=1).std().shift(1)
        
        # Fill NaN values that occur at the beginning
        if is_training:
            stats[f'{col}_roll_mean_3_median'] = processed_df[f'{col}_roll_mean_3'].median()
            stats[f'{col}_roll_mean_7_median'] = processed_df[f'{col}_roll_mean_7'].median()
            stats[f'{col}_roll_std_7_median'] = processed_df[f'{col}_roll_std_7'].median()
            
            processed_df[f'{col}_roll_mean_3'].fillna(stats[f'{col}_roll_mean_3_median'], inplace=True)
            processed_df[f'{col}_roll_mean_7'].fillna(stats[f'{col}_roll_mean_7_median'], inplace=True)
            processed_df[f'{col}_roll_std_7'].fillna(stats[f'{col}_roll_std_7_median'], inplace=True)
        else:
            processed_df[f'{col}_roll_mean_3'].fillna(reference_data[f'{col}_roll_mean_3_median'], inplace=True)
            processed_df[f'{col}_roll_mean_7'].fillna(reference_data[f'{col}_roll_mean_7_median'], inplace=True)
            processed_df[f'{col}_roll_std_7'].fillna(reference_data[f'{col}_roll_std_7_median'], inplace=True)
        
        # Calculate difference between current value and rolling mean
        processed_df[f'{col}_diff_mean_3'] = processed_df[col] - processed_df[f'{col}_roll_mean_3']
        processed_df[f'{col}_diff_mean_7'] = processed_df[col] - processed_df[f'{col}_roll_mean_7']
        
        # Z-score compared to rolling history
        processed_df[f'{col}_zscore'] = (processed_df[col] - processed_df[f'{col}_roll_mean_7']) / (processed_df[f'{col}_roll_std_7'] + 0.001)
    
    # 8. Day of year cyclical features
    processed_df['day_sin'] = np.sin(2 * np.pi * processed_df['day'] / 365)
    processed_df['day_cos'] = np.cos(2 * np.pi * processed_df['day'] / 365)
    
    # 9. Add interaction terms for common weather patterns
    processed_df['temp_humidity'] = processed_df['temparature'] * processed_df['humidity']
    processed_df['wind_temp'] = processed_df['windspeed'] * processed_df['temparature']
    
    return processed_df, stats

def build_model(X_train, y_train, X_val=None, y_val=None, use_validation=True):
    """
    Build and train models with proper evaluation
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_val, y_val : validation data (optional)
    use_validation : bool
        Whether to use validation data for evaluation
    
    Returns:
    --------
    model_data : dict
        Dictionary containing all model-related objects
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if use_validation and X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    
    # Handle class imbalance if needed - but only for training
    class_counts = np.bincount(y_train)
    
    # Only use SMOTEENN if significant class imbalance exists
    if min(class_counts) / max(class_counts) < 0.25:
        smoteenn = SMOTEENN(random_state=42)
        X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scaled, y_train)
        print(f"Applied SMOTEENN: Original shape: {X_train_scaled.shape}, Resampled shape: {X_train_resampled.shape}")
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
        print("Class imbalance not severe enough to require SMOTEENN")
    
    # Define less complex models to reduce overfitting
    base_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, 
            C=0.01,  # More regularization 
            class_weight='balanced', 
            penalty='l2'
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,  # Fewer trees 
            max_depth=8,  # Less depth to reduce overfitting
            min_samples_split=10, 
            min_samples_leaf=5,
            max_features='sqrt',  # Use only sqrt(n_features) to reduce overfitting
            class_weight='balanced',
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.01,  # Slower learning
            max_depth=3,  # Much reduced depth
            subsample=0.7,
            max_features='sqrt',
            random_state=42
        )
    }
    
    # Function to evaluate models
    def evaluate_model(model, name):
        model.fit(X_train_resampled, y_train_resampled)
        
        if use_validation and X_val is not None:
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            print(f"{name} - Validation ROC AUC: {auc:.4f}")
        else:
            # Use cross-validation for better model evaluation
            from sklearn.model_selection import cross_val_score
            auc = np.mean(cross_val_score(model, X_train_resampled, y_train_resampled, 
                                          cv=5, scoring='roc_auc'))
            print(f"{name} - Cross-val ROC AUC: {auc:.4f}")
        
        return model, auc
    
    # Train and evaluate base models
    trained_models = {}
    scores = {}
    
    for name, model in base_models.items():
        trained_model, score = evaluate_model(model, name)
        trained_models[name] = trained_model
        scores[name] = score
    
    # Create a simple ensemble with the two best models only
    best_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    print(f"Best models: {[model[0] for model in best_models]}")
    
    voting_clf = VotingClassifier(
        estimators=[
            (name, trained_models[name]) for name, _ in best_models
        ],
        voting='soft'
    )
    
    # Train the ensemble
    voting_clf.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the ensemble
    if use_validation and X_val is not None:
        y_ensemble_proba = voting_clf.predict_proba(X_val_scaled)[:, 1]
        ensemble_auc = roc_auc_score(y_val, y_ensemble_proba)
        print(f"Ensemble Validation ROC AUC: {ensemble_auc:.4f}")
    else:
        from sklearn.model_selection import cross_val_score
        ensemble_auc = np.mean(cross_val_score(voting_clf, X_train_resampled, y_train_resampled, 
                                                cv=5, scoring='roc_auc'))
        print(f"Ensemble Cross-val ROC AUC: {ensemble_auc:.4f}")
    
    # Package everything needed for prediction
    model_data = {
        'scaler': scaler,
        'base_models': trained_models,
        'ensemble_model': voting_clf,
        'feature_list': list(X_train.columns),
        'best_model_type': 'voting_ensemble',
        'ensemble_auc': ensemble_auc
    }
    
    return model_data

def train_model_with_proper_validation():
    """Main function to train the model with proper validation"""
    # Load the data
    df = pd.read_csv('data/train.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Process the data
    processed_df, stats = preprocess_data(df, is_training=True)
    
    # Print class distribution
    print("Class distribution:")
    print(processed_df['rainfall'].value_counts(normalize=True))
    
    # Select features and target
    X = processed_df.drop(['id', 'day', 'rainfall', 'month'], axis=1)  # Remove original month after creating cyclical features
    y = processed_df['rainfall']
    
    # Use time series cross-validation instead of random split
    # This respects the temporal nature of the data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Get the last split for final validation
    train_indices = None
    val_indices = None
    
    for train_index, val_index in tscv.split(X):
        train_indices = train_index
        val_indices = val_index
    
    X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
    
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    
    # Build the model
    model_data = build_model(X_train, y_train, X_val, y_val, use_validation=True)
    
    # Add stats to model data
    model_data['stats'] = stats
    
    # Save model pipeline
    with open('rainfall_prediction_pipeline.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model trained and saved to 'rainfall_prediction_pipeline.pkl'")
    return model_data

def predict_rainfall(new_data_path):
    """
    Make predictions on new data
    
    Parameters:
    -----------
    new_data_path : str
        Path to the new data CSV file
    
    Returns:
    --------
    submission : pandas DataFrame
        DataFrame with id and rainfall probability
    """
    # Load the model pipeline
    with open('rainfall_prediction_pipeline.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    # Save IDs
    ids = new_data['id'].copy()
    
    # Process the new data using the same transformations
    processed_data, _ = preprocess_data(new_data, is_training=False, reference_data=model_data['stats'])
    
    # Select only the needed features
    feature_list = model_data['feature_list']
    
    # Check for missing columns and add them
    for col in feature_list:
        if col not in processed_data.columns:
            processed_data[col] = 0  # Default value
    
    # Select only the needed features
    X_new = processed_data[feature_list]
    
    # Scale the features
    X_new_scaled = model_data['scaler'].transform(X_new)
    
    # Make predictions with the ensemble model
    predictions = model_data['ensemble_model'].predict_proba(X_new_scaled)[:, 1]
    
    # Create submission file
    submission = pd.DataFrame({
        'id': ids,
        'rainfall': predictions
    })
    
    return submission

# Main execution
if __name__ == "__main__":
    print("Training model with proper validation...")
    model_data = train_model_with_proper_validation()
    
    # Predict on test data
    print("Making predictions on test data...")
    test_file = 'data/test.csv'
    submission = predict_rainfall(test_file)
    submission.to_csv('improved_rainfall_predictions.csv', index=False)
    print("Predictions saved to 'improved_rainfall_predictions.csv'")
    

