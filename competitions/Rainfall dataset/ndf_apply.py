import pandas as pd
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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

class NeuralDecisionForest(nn.Module):
    def __init__(self, input_dim, num_trees=10, depth=5):
        super(NeuralDecisionForest, self).__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.depth = depth
        self.num_leaves = 2 ** depth
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Decision paths (internal nodes)
        self.decision_nodes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 1)
            ) for _ in range(num_trees * (2**depth - 1))
        ])
        
        # Leaf node probabilities
        self.leaf_probabilities = nn.Parameter(torch.randn(num_trees, self.num_leaves))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Initialize probabilities
        tree_outputs = torch.zeros(batch_size, self.num_trees, dtype=x.dtype, device=x.device)
        
        for tree_idx in range(self.num_trees):
            # Start with equal probabilities for reaching each leaf
            leaf_probs = torch.ones(batch_size, self.num_leaves, dtype=x.dtype, device=x.device) / self.num_leaves
            
            # Compute probabilities for each decision path
            for level in range(self.depth):
                for node_idx in range(2**level - 1, 2**(level+1) - 1):
                    # Calculate decision probability for this node
                    node_global_idx = tree_idx * (2**self.depth - 1) + node_idx
                    decision = torch.sigmoid(self.decision_nodes[node_global_idx](features))
                    
                    # Update leaf probabilities based on this decision
                    for leaf_idx in range(self.num_leaves):
                        # Determine whether this leaf is in the left or right subtree
                        # of the current node
                        direction = (leaf_idx >> (self.depth - level - 1)) & 1
                        
                        if direction == 0:  # Left
                            leaf_probs[:, leaf_idx] *= (1 - decision.squeeze())
                        else:  # Right
                            leaf_probs[:, leaf_idx] *= decision.squeeze()
            
            # Compute the final probability for each tree as a weighted sum of leaf predictions
            leaf_preds = torch.sigmoid(self.leaf_probabilities[tree_idx])
            tree_outputs[:, tree_idx] = torch.sum(leaf_probs * leaf_preds, dim=1)
        
        # Average predictions from all trees
        final_output = torch.mean(tree_outputs, dim=1)
        
        return final_output.unsqueeze(1)

def train_neural_forest_binary(X, y, input_dim, num_trees=10, depth=5, batch_size=64, 
                               epochs=50, learning_rate=0.001, validation_split=0.2):
    """
    Train a Neural Decision Forest for binary classification
    
    Parameters:
    -----------
    X : torch.Tensor
        Input features tensor
    y : torch.Tensor
        Target tensor
    input_dim : int
        Number of input features
    num_trees : int
        Number of trees in the forest
    depth : int
        Depth of each tree
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    validation_split : float
        Proportion of data to use for validation
    
    Returns:
    --------
    model : NeuralDecisionForest
        Trained model
    metrics : dict
        Dictionary containing training metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Split data into training and validation
    val_size = int(len(X) * validation_split)
    indices = torch.randperm(len(X))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = NeuralDecisionForest(input_dim, num_trees, depth).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_dataset)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(batch_y.cpu().numpy().flatten())
        
        val_loss /= len(val_dataset)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {train_loss:.4f} - '
                  f'Val Loss: {val_loss:.4f} - '
                  f'Val AUC: {val_auc:.4f}')
    
    # Final validation metrics
    model.eval()
    all_val_preds = []
    all_val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            outputs = model(batch_X)
            all_val_preds.extend(outputs.cpu().numpy().flatten())
            all_val_targets.extend(batch_y.cpu().numpy().flatten())
    
    final_val_auc = roc_auc_score(all_val_targets, all_val_preds)
    print(f'Final Validation AUC: {final_val_auc:.4f}')
    
    metrics = {
        'history': history,
        'final_val_auc': final_val_auc
    }
    
    return model, metrics

def train_model_with_ndf():
    """Main function to train the model with Neural Decision Forest"""
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
    
    # Use TimeSeriesSplit to respect temporal nature of data
    tscv = TimeSeriesSplit(n_splits=5)
    train_indices = None
    val_indices = None
    
    for train_index, val_index in tscv.split(X):
        train_indices = train_index
        val_indices = val_index
    
    X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
    
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    
    # Train NDF model
    ndf_model, metrics = train_neural_forest_binary(
        X_tensor, y_tensor, 
        input_dim=X.shape[1],
        batch_size=128,
        epochs=30,
        learning_rate=0.001,
        validation_split=0.2
    )
    
    # Save model and related data
    model_data = {
        'scaler': scaler,
        'ndf_model': ndf_model,
        'stats': stats,
        'feature_list': list(X.columns),
        'metrics': metrics
    }
    
    with open('ndf_rainfall_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("NDF model trained and saved to 'ndf_rainfall_model.pkl'")
    return model_data

def predict_rainfall(new_data_path):
    """
    Make predictions on new data using NDF model
    
    Parameters:
    -----------
    new_data_path : str
        Path to the new data CSV file
    
    Returns:
    --------
    submission : pandas DataFrame
        DataFrame with id and rainfall probability
    """
    # Load the model data
    with open('ndf_rainfall_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Get the model and scaler
    ndf_model = model_data['ndf_model']
    scaler = model_data['scaler']
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    # Save IDs
    ids = new_data['id'].copy()
    
    # Process the new data
    processed_data, _ = preprocess_data(new_data, is_training=False, reference_data=model_data['stats'])
    
    # Select features
    feature_list = model_data['feature_list']
    
    # Check for missing columns and add them
    for col in feature_list:
        if col not in processed_data.columns:
            processed_data[col] = 0
    
    X_new = processed_data[feature_list]
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Convert to tensor
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
    
    # Set model to evaluation mode
    ndf_model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = ndf_model(X_new_tensor).numpy().flatten()
    
    # Create submission file
    submission = pd.DataFrame({
        'id': ids,
        'rainfall': predictions
    })
    
    return submission

def ndf_predict(ndf):
    """
    Make predictions using the NDF model on a new DataFrame
    
    Parameters:
    -----------
    ndf : pandas DataFrame
        New data in the same format as the training data
    
    Returns:
    --------
    predictions : pandas DataFrame
        DataFrame with id and rainfall probability predictions
    """
    # Load the model data
    with open('ndf_rainfall_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Get the model and scaler
    ndf_model = model_data['ndf_model']
    scaler = model_data['scaler']
    
    # Save IDs
    ids = ndf['id'].copy()
    
    # Process the data
    processed_data, _ = preprocess_data(ndf, is_training=False, reference_data=model_data['stats'])
    
    # Select features
    feature_list = model_data['feature_list']
    
    # Check for missing columns and add them
    for col in feature_list:
        if col not in processed_data.columns:
            processed_data[col] = 0
    
    X_new = processed_data[feature_list]
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Convert to tensor
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
    
    # Set model to evaluation mode
    ndf_model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = ndf_model(X_new_tensor).numpy().flatten()
    
    # Create predictions DataFrame
    result = pd.DataFrame({
        'id': ids,
        'rainfall': predictions
    })
    
    return result

# Main execution
if __name__ == "__main__":
    # Train model
    print("Training Neural Decision Forest model...")
    model_data = train_model_with_ndf()
    
    # Predict on test data
    print("Making predictions on test data...")
    test_file = 'data/test.csv'
    submission = predict_rainfall(test_file)
    submission.to_csv('ndf_rainfall_predictions.csv', index=False)
    print("Predictions saved to 'ndf_rainfall_predictions.csv'")
    