import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import warnings
warnings.filterwarnings('ignore')

TEST = True

    
# Load the data
df = pd.read_csv('data/train.csv')  # Assuming this is the full dataset

# Exploratory Data Analysis
print(f"Dataset shape: {df.shape}")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values per column:\n{missing_values}")

# Handle missing values if any
df = df.fillna(df.median())

# Feature Engineering
# Adding potentially useful features based on domain knowledge
df['temp_range'] = df['maxtemp'] - df['mintemp']
df['humidity_pressure_ratio'] = df['humidity'] / df['pressure']
df['is_windy'] = (df['windspeed'] > 30).astype(int)

# Visualize relationship between key features and rainfall
plt.figure(figsize=(15, 10))
features_to_plot = ['humidity', 'cloud', 'pressure', 'temparature', 'windspeed', 'temp_range']
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='rainfall', y=feature, data=df)
    plt.title(f'{feature} vs Rainfall')
plt.tight_layout()
plt.savefig('feature_vs_rainfall.png')

# Select features and target
X = df.drop(['id', 'day', 'rainfall'], axis=1)  # Assuming these are the non-feature columns
y = df['rainfall']

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Class imbalance check
class_counts = df['rainfall'].value_counts(normalize=True)
print(f"Class distribution: {class_counts}")

if not TEST:

    # Model Training with Cross-Validation
    # 1. Logistic Regression
    logreg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    logreg_params = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__class_weight': [None, 'balanced']
    }

    # 2. Random Forest
    rf_pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    rf_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__class_weight': [None, 'balanced']
    }

    # 3. Gradient Boosting
    gb_pipeline = Pipeline([
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])

    gb_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    }

    # Create stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train and evaluate models
    def train_evaluate_model(pipeline, params, name):
        grid_search = GridSearchCV(
            pipeline, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"\n{name} Results:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Evaluate on validation set
        y_val_pred = grid_search.predict(X_val)
        y_val_prob = grid_search.predict_proba(X_val)[:, 1]
        
        print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
        print(f"Validation ROC-AUC: {roc_auc_score(y_val, y_val_prob):.4f}")
        print(f"Validation Classification Report:\n{classification_report(y_val, y_val_pred)}")
        
        return grid_search

    # Train all models
    logreg_model = train_evaluate_model(logreg_pipeline, logreg_params, "Logistic Regression")
    rf_model = train_evaluate_model(rf_pipeline, rf_params, "Random Forest")
    gb_model = train_evaluate_model(gb_pipeline, gb_params, "Gradient Boosting")

    # Select best model based on validation performance
    models = [logreg_model, rf_model, gb_model]
    model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    val_scores = []

    for model in models:
        val_scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

    best_model_index = np.argmax(val_scores)
    best_model = models[best_model_index]
    best_model_name = model_names[best_model_index]

    print(f"\nBest model is {best_model_name} with validation ROC-AUC of {val_scores[best_model_index]:.4f}")

    # Evaluate best model on test set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    print(f"\nTest Set Evaluation for {best_model_name}:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
    print(f"Test Classification Report:\n{classification_report(y_test, y_test_pred)}")

    # Feature importance analysis
    if best_model_name == "Logistic Regression":
        coefs = best_model.best_estimator_.named_steps['classifier'].coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(coefs)
        }).sort_values('Importance', ascending=False)
    elif best_model_name in ["Random Forest", "Gradient Boosting"]:
        importances = best_model.best_estimator_.named_steps['classifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('confusion_matrix.png')

    # Save the best model
 
    with open('rainfall_prediction_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print("\nModel saved as 'rainfall_prediction_model.pkl'")

else:
    # Predict on new data 'data/test.csv'
    # Load new data for prediction
    test_data = pd.read_csv('data/test.csv')

    # Preprocess the new data
    test_data['temp_range'] = test_data['maxtemp'] - test_data['mintemp']
    test_data['humidity_pressure_ratio'] = test_data['humidity'] / test_data['pressure']
    test_data['is_windy'] = (test_data['windspeed'] > 30).astype(int)

    # Handle missing values if any
    test_data = test_data.fillna(test_data.median())

    # Select features (ensure the same features as used in training)
    X_new = test_data.drop(['id', 'day'], axis=1)

    # Scale the features
    X_new_scaled = scaler.transform(X_new)

    # Reload the saved model
    with open('rainfall_prediction_model.pkl', 'rb') as f:
        saved_model = pickle.load(f)

    # Make predictions
    new_probabilities = saved_model.predict_proba(X_new_scaled)[:, 1]

    # Save predictions to a CSV file
    output = pd.DataFrame({
        'id': test_data['id'],
        'rainfall': new_probabilities
    })
    output.to_csv('rainfall_predictions.csv', index=False)

    print("\nPredictions saved to 'rainfall_predictions.csv'")