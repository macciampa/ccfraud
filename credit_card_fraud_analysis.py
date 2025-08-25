import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data():
    """Load and prepare the credit card fraud dataset."""
    print("Loading credit card fraud dataset...")
    
    # Load the dataset
    df = pd.read_csv('creditcard.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")  # All except 'Class'
    print(f"Target: Class")
    
    # Check class distribution
    class_counts = df['Class'].value_counts()
    print(f"\nClass distribution:")
    print(f"Legitimate transactions (0): {class_counts[0]:,}")
    print(f"Fraudulent transactions (1): {class_counts[1]:,}")
    print(f"Fraud percentage: {class_counts[1] / len(df) * 100:.2f}%")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y

def split_data(X, y):
    """Split data into training, validation, and test sets."""
    print("\nSplitting data into train/validation/test sets...")
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: 75% train, 25% validation (of the remaining 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler."""
    print("\nScaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_models(X_train, X_val, y_train, y_val):
    """Train all four models."""
    print("\nTraining models...")
    
    models = {}
    
    # 1. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 2. Gradient Boosting Decision Tree (GBDT)
    print("Training GBDT...")
    gbdt = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    gbdt.fit(X_train, y_train)
    models['GBDT'] = gbdt
    
    # 3. Logistic Regression (needs scaled features)
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 4. Neural Network (needs scaled features)
    print("Training Neural Network...")
    nn = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn.fit(X_train, y_train)
    models['Neural Network'] = nn
    
    return models

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'auc_pr': auc_pr,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance for models that support it."""
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_[0])
    else:
        # Neural network - no direct feature importance
        return None
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def plot_results(results):
    """Create comprehensive visualization of results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Credit Card Fraud Detection - Model Comparison', fontsize=16)
    
    # 1. Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auc_pr']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (model_name, result) in enumerate(results.items()):
        values = [result[metric] for metric in metrics]
        axes[0, 0].bar(x + i * width, values, width, label=model_name, alpha=0.8)
    
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(metric_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curves
    for model_name, result in results.items():
        axes[0, 1].plot(result['fpr'], result['tpr'], label=f"{model_name} (AUC={result['auc']:.3f})")
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curves
    for model_name, result in results.items():
        axes[0, 2].plot(result['recall_curve'], result['precision_curve'], 
                       label=f"{model_name} (AUC-PR={result['auc_pr']:.3f})")
    
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curves')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4-6. Confusion Matrices
    for i, (model_name, result) in enumerate(results.items()):
        row = 1
        col = i
        if col >= 3:
            break
            
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
        axes[row, col].set_title(f'{model_name} - Confusion Matrix')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(feature_importance_dict):
    """Plot feature importance for models that support it."""
    if not feature_importance_dict:
        print("No feature importance available for plotting.")
        return
    
    n_models = len(feature_importance_dict)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Importance Comparison', fontsize=16)
    
    axes = axes.flatten()
    
    for i, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
        if importance_df is None:
            continue
            
        # Plot top 15 features
        top_features = importance_df.head(15)
        axes[i].barh(range(len(top_features)), top_features['importance'])
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features['feature'])
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'{model_name} - Top 15 Features')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_results_summary(results, y_test):
    """Print a comprehensive summary of all results."""
    print("\n" + "="*80)
    print("CREDIT CARD FRAUD DETECTION - MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Create summary table
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'AUC-ROC': f"{result['auc']:.4f}",
            'AUC-PR': f"{result['auc_pr']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nPerformance Metrics Summary:")
    print(summary_df.to_string(index=False))
    
    # Print detailed results for each model
    for model_name, result in results.items():
        print(f"\n{'-'*60}")
        print(f"DETAILED RESULTS FOR {model_name.upper()}")
        print(f"{'-'*60}")
        
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1-Score: {result['f1']:.4f}")
        print(f"AUC-ROC: {result['auc']:.4f}")
        print(f"AUC-PR: {result['auc_pr']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(result['confusion_matrix'])
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, result['y_pred']))

def main():
    """Main execution function."""
    print("Credit Card Fraud Detection - Model Comparison")
    print("="*50)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Scale features for models that need it
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Train models (using scaled features for all models for consistency)
    models = train_models(X_train_scaled, X_val_scaled, y_train, y_val)
    
    # Evaluate all models
    print("\nEvaluating models on test set...")
    results = {}
    feature_importance_dict = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        result = evaluate_model(model, X_test_scaled, y_test, model_name)
        results[model_name] = result
        
        # Get feature importance
        feature_importance = get_feature_importance(model, X.columns, model_name)
        feature_importance_dict[model_name] = feature_importance
    
    # Print results summary
    print_results_summary(results, y_test)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_results(results)
    plot_feature_importance(feature_importance_dict)
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1'],
            'AUC_ROC': result['auc'],
            'AUC_PR': result['auc_pr']
        })
    
    results_df = pd.DataFrame(summary_data)
    results_df.to_csv('model_comparison_results.csv', index=False)
    
    # Save feature importance to CSV
    for model_name, importance_df in feature_importance_dict.items():
        if importance_df is not None:
            importance_df.to_csv(f'{model_name.replace(" ", "_")}_feature_importance.csv', index=False)
    
    print("\nAnalysis complete! Results saved to:")
    print("- model_comparison_results.csv")
    print("- model_comparison_results.png")
    print("- feature_importance.png")
    print("- Individual feature importance CSV files")

if __name__ == "__main__":
    main() 