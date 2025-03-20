import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

class DeepNeuralFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128, 256, 128], activation='relu', dropout_rate=0.2):
        super(DeepNeuralFeatureExtractor, self).__init__()
        
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'selu':
            self.activation = F.selu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)  # Swish activation
        else:
            self.activation = F.relu  # Default to ReLU
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers with skip connections
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i]))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            
            # Skip connection if dimensions match or create projection
            if i >= 2:  # Add skip connections from the 2nd layer onwards
                if hidden_dims[i-2] == hidden_dims[i]:
                    self.skip_connections.append(nn.Identity())
                else:
                    self.skip_connections.append(nn.Linear(hidden_dims[i-2], hidden_dims[i]))
    
    def forward(self, x):
        prev_outputs = []
        
        # First layer
        x = self.layers[0](x)
        x = self.batch_norms[0](x)
        x = self.activation(x)
        x = self.dropout_layers[0](x)
        prev_outputs.append(x)
        
        # Second layer (no skip connection yet)
        x = self.layers[1](x)
        x = self.batch_norms[1](x)
        x = self.activation(x)
        x = self.dropout_layers[1](x)
        prev_outputs.append(x)
        
        # Remaining layers with skip connections
        for i in range(2, len(self.layers)):
            skip_input = prev_outputs[i-2]
            skip_output = self.skip_connections[i-2](skip_input)
            
            current_output = self.layers[i](x)
            current_output = self.batch_norms[i](current_output)
            x = self.activation(current_output + skip_output)
            x = self.dropout_layers[i](x)
            prev_outputs.append(x)
            
        return x

class NeuralDecisionTree(nn.Module):
    def __init__(self, input_dim, output_dim, depth=4, activation='tanh'):
        super(NeuralDecisionTree, self).__init__()
        
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.output_dim = output_dim
        
        # Set activation function for decision nodes
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            self.activation = torch.tanh  # Default to tanh
        
        # Decision nodes - these are essentially routing functions
        # Each decision node outputs the probability of routing to the left
        self.decision_nodes = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2**depth - 1)
        )
        
        # Leaf node probabilities
        self.leaf_probabilities = nn.Parameter(torch.randn(self.n_leaf, output_dim))
    
    def forward(self, x):
        """
        Forward pass through the Neural Decision Tree
        
        Args:
            x: Input features, shape [batch_size, input_dim]
            
        Returns:
            Predicted class probabilities, shape [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Decision node outputs: probability of going left at each node
        decision = self.activation(self.decision_nodes(x))  # [batch_size, 2^depth - 1]
        
        # Calculate the probability of reaching each leaf node
        mu = self._compute_mu(batch_size, decision)  # [batch_size, 2^depth]
        
        # Getting the final prediction from leaf node probabilities
        outputs = torch.matmul(mu, F.softmax(self.leaf_probabilities, dim=1))  # [batch_size, output_dim]
        
        return outputs
    
    def _compute_mu(self, batch_size, decision):
        """
        Compute the probability of reaching each leaf node for each sample
        
        Args:
            batch_size: Number of samples in the batch
            decision: Decision probabilities at each node [batch_size, 2^depth - 1]
            
        Returns:
            Probability of reaching each leaf for each sample [batch_size, 2^depth]
        """
        mu = torch.ones((batch_size, 1), device=decision.device)
        
        # Initialize list to keep track of probabilities
        begin_idx = 0
        end_idx = 1
        
        # Traverse through the tree level by level
        for level in range(self.depth):
            # Get decision probabilities for this level
            level_decisions = decision[:, begin_idx:end_idx]  # [batch_size, 2^level]
            
            # Left and right probabilities
            level_mu_left = mu * level_decisions
            level_mu_right = mu * (1 - level_decisions)
            
            # Update mu by concatenating left and right probabilities
            mu = torch.cat((level_mu_left, level_mu_right), dim=1)
            
            # Update indices for the next level
            begin_idx = end_idx
            end_idx = begin_idx + 2**(level+1)
            
        return mu

class NeuralDecisionForest(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim=128, n_trees=5, tree_depth=4, 
                 nn_activation='relu', tree_activation='tanh', dropout_rate=0.2):
        super(NeuralDecisionForest, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.n_trees = n_trees
        
        # Feature extractor: Deep neural network with skip connections
        self.feature_extractor = DeepNeuralFeatureExtractor(
            input_dim, 
            hidden_dims=[64, 128, 256, feature_dim],
            activation=nn_activation,
            dropout_rate=dropout_rate
        )
        
        # Forest: collection of decision trees
        self.trees = nn.ModuleList([
            NeuralDecisionTree(feature_dim, output_dim, depth=tree_depth, activation=tree_activation)
            for _ in range(n_trees)
        ])
        
    def forward(self, x):
        """
        Forward pass through the Neural Decision Forest
        
        Args:
            x: Raw input features [batch_size, input_dim]
            
        Returns:
            Predicted class probabilities [batch_size, output_dim]
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Get predictions from each tree
        tree_outputs = []
        for tree in self.trees:
            tree_outputs.append(tree(features))
        
        # Average the predictions
        outputs = torch.stack(tree_outputs, dim=0).mean(dim=0)
        
        return outputs

def evaluate_binary_classifier(model, X, y, threshold=0.5):
    """
    Evaluate a binary classifier with AUC-ROC and related metrics
    
    Args:
        model: Trained model
        X: Input features tensor
        y: True binary labels tensor
        threshold: Decision threshold for binary classification
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        # Get raw prediction scores (probabilities)
        outputs = model(X)
        
        # For binary classification, get the probability of the positive class
        if outputs.shape[1] == 2:
            # If model outputs two probabilities (multi-class format)
            y_score = outputs[:, 1].cpu().numpy()
        else:
            # If model outputs a single probability (binary format)
            y_score = outputs.squeeze().cpu().numpy()
        
        # Convert labels to numpy
        y_true = y.cpu().numpy()
        
        # Calculate AUC-ROC
        auc_score = roc_auc_score(y_true, y_score)
        
        # Get ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Calculate binary predictions using threshold
        y_pred = (y_score >= threshold).astype(int)
        
        # Calculate accuracy
        accuracy = (y_pred == y_true).mean()
        
        # Calculate precision, recall, and F1 score
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()
        false_positives = ((y_pred == 1) & (y_true == 0)).sum()
        false_negatives = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Return all metrics
        metrics = {
            'auc_roc': auc_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }
        }
        
        return metrics

def plot_roc_curve(metrics, title='Receiver Operating Characteristic'):
    """
    Plot ROC curve from evaluation metrics
    
    Args:
        metrics: Dictionary containing ROC curve data
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(
        metrics['roc_curve']['fpr'], 
        metrics['roc_curve']['tpr'], 
        color='darkorange', 
        lw=2, 
        label=f'ROC curve (area = {metrics["auc_roc"]:.3f})'
    )
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    # Add metrics text
    metrics_text = (
        f"AUC: {metrics['auc_roc']:.3f}\n"
        f"Accuracy: {metrics['accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall: {metrics['recall']:.3f}\n"
        f"F1 Score: {metrics['f1_score']:.3f}"
    )
    plt.text(0.7, 0.3, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True)
    plt.show()

# Parameterized training function for binary classification
def train_neural_forest_binary(
    X=None, 
    y=None, 
    input_dim=20, 
    feature_dim=128, 
    n_trees=5, 
    tree_depth=4,
    nn_activation='relu',
    tree_activation='tanh',
    dropout_rate=0.2,
    lr=0.01,
    epochs=10,
    batch_size=32,
    seed=42,
    verbose=False,
    plot_roc=False
):
    """
    Train a Neural Decision Forest model specifically for binary classification
    with AUC-ROC evaluation
    
    Args:
        X: Input features tensor (if None, synthetic data will be generated)
        y: Target binary labels tensor (if None, synthetic data will be generated)
        input_dim: Dimension of input features
        feature_dim: Dimension of features extracted by neural network
        n_trees: Number of decision trees in the forest
        tree_depth: Depth of each decision tree
        nn_activation: Activation function for neural network
        tree_activation: Activation function for decision trees
        dropout_rate: Dropout probability for neural network
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Mini-batch size
        seed: Random seed for reproducibility
        verbose: Whether to print training progress
        plot_roc: Whether to plot ROC curve after training
        
    Returns:
        Trained NeuralDecisionForest model, training history, and evaluation metrics
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Output dimension is always 1 for binary classification
    output_dim = 1
    
    # Create synthetic data if not provided
    if X is None or y is None:
        if verbose:
            print("Generating synthetic binary classification data...")
        
        # Generate balanced binary classification data
        n_samples = 1000
        X = torch.randn(n_samples, input_dim)
        
        # Create synthetic binary labels
        # Using a non-linear decision boundary for more realistic data
        z = torch.sum(X[:, :5]**2, dim=1) - torch.sum(X[:, 5:10], dim=1)
        y = (torch.sigmoid(z) > 0.5).float()
    
    num_samples = X.shape[0]
    
    # Ensure y is binary and proper shape
    assert torch.all((y == 0) | (y == 1)), "Labels must be binary (0 or 1)"
    if y.dim() > 1:
        y = y.squeeze()
    
    # Create model
    model = NeuralDecisionForest(
        input_dim=input_dim,
        output_dim=output_dim,  # Binary classification
        feature_dim=feature_dim,
        n_trees=n_trees,
        tree_depth=tree_depth,
        nn_activation=nn_activation,
        tree_activation=tree_activation,
        dropout_rate=dropout_rate
    )
    
    # Loss and optimizer for binary classification
    criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid activation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    # Split data into train and validation sets (80/20)
    train_size = int(0.8 * num_samples)
    indices = torch.randperm(num_samples)
    
    X_train = X[indices[:train_size]]
    y_train = y[indices[:train_size]]
    X_val = X[indices[train_size:]]
    y_val = y[indices[train_size:]]
    
    # Training loop
    best_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle training data
        train_indices = torch.randperm(train_size)
        X_train_shuffled = X_train[train_indices]
        y_train_shuffled = y_train[train_indices]
        
        # Mini-batch training
        epoch_loss = 0.0
        for i in range(0, train_size, batch_size):
            # Get mini-batch
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Forward pass
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
        
        # Calculate training loss
        train_loss = epoch_loss / train_size
        history['loss'].append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val).item()
            history['val_loss'].append(val_loss)
            
            # Calculate AUC-ROC on validation set
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_true = y_val.cpu().numpy()
            val_auc = roc_auc_score(val_true, val_probs)
            history['val_auc'].append(val_auc)
            
            # Save best model based on validation AUC
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict().copy()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val AUC: {val_auc:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    metrics = evaluate_binary_classifier(model, X_val, y_val)
    
    if verbose:
        print("\nFinal Evaluation Metrics:")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Plot ROC curve
    if plot_roc:
        plot_roc_curve(metrics)
    
    return model, history, metrics

# Example for binary classification with AUC-ROC evaluation
def run_binary_example():
    """
    Run a complete example of binary classification with the Neural Decision Forest
    """
    # Generate non-linear binary classification data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    n_features = 20
    
    # Create a challenging non-linear dataset
    X = torch.randn(n_samples, n_features)
    
    # Non-linear decision boundary
    w1 = torch.randn(5)
    w2 = torch.randn(5)
    z = torch.sum(X[:, :5] * w1, dim=1) - torch.sum(X[:, 5:10]**2 * w2, dim=1) + torch.sin(torch.sum(X[:, 10:15], dim=1))
    prob = torch.sigmoid(z)
    y = (prob > 0.5).float()
    
    # Add some noise
    flip_indices = torch.randperm(n_samples)[:int(0.05 * n_samples)]
    y[flip_indices] = 1 - y[flip_indices]
    
    # Train the model
    model, history, metrics = train_neural_forest_binary(
        X=X,
        y=y,
        input_dim=n_features,
        feature_dim=64,
        n_trees=10,
        tree_depth=5,
        nn_activation='gelu',
        tree_activation='tanh',
        dropout_rate=0.3,
        lr=0.005,
        epochs=50,
        batch_size=64,
        seed=42,
        verbose=True,
        plot_roc=True
    )
    
    return model, history, metrics

# Uncomment to run the example
# model, history, metrics = run_binary_example()