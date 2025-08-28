import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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

# Note: This script has an option to use class weights and sample weights to handle the severe class imbalance
# in the credit card fraud dataset. The imbalance ratio is approximately 577:1 (legitimate:fraud).
# Currently set to use_class_weights=False for standard training without class imbalance handling.

def load_and_prepare_data():
    """Load and prepare the credit card fraud dataset."""
    print("Loading credit card fraud dataset...")
    
    # Load the dataset
    df = pd.read_csv('creditcard.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")  # All except 'Class'
    print(f"Target: Class")
    
    # Check class distribution and imbalance
    class_counts = df['Class'].value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]
    
    print(f"\nClass distribution:")
    print(f"Legitimate transactions (0): {class_counts[0]:,}")
    print(f"Fraudulent transactions (1): {class_counts[1]:,}")
    print(f"Fraud percentage: {class_counts[1] / len(df) * 100:.2f}%")
    print(f"Imbalance ratio (legitimate:fraud): {imbalance_ratio:.1f}:1")
    print(f"Severity: {'Severe' if imbalance_ratio > 100 else 'Moderate' if imbalance_ratio > 10 else 'Mild'} imbalance")
    
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

def train_models(X_train, X_val, y_train, y_val, use_class_weights=False):
    """Train all four models with optional class weights to handle imbalance."""
    if use_class_weights:
        print("\nTraining models with class weights to handle imbalance...")
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print(f"Class weights: {class_weight_dict}")
    else:
        print("\nTraining models without class weights (standard training)...")
    
    models = {}
    
    # 1. Random Forest
    print("Training Random Forest...")
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if use_class_weights:
        rf_params['class_weight'] = 'balanced'
    
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 2. XGBoost Classifier
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    if use_class_weights:
        # XGBoost supports sample_weight for handling class imbalance
        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 1] = class_weights[1]  # Fraud class weight
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        xgb.fit(X_train, y_train)
    
    models['XGBoost'] = xgb
    
    # 3. Logistic Regression (needs scaled features)
    print("Training Logistic Regression...")
    lr_params = {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'liblinear'
    }
    
    if use_class_weights:
        lr_params['class_weight'] = 'balanced'
    
    lr = LogisticRegression(**lr_params)
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
    
    if use_class_weights:
        # Neural Network doesn't support class_weight, so we'll use sample_weight
        nn.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        nn.fit(X_train, y_train)
    
    models['Neural Network'] = nn
    
    # Report model sizes
    print("\n" + "="*60)
    print("MODEL SIZE ANALYSIS")
    print("="*60)
    
    for model_name, model in models.items():
        model_size = get_model_size(model)
        print(f"{model_name}: {model_size}")
    
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
    
    # Additional metrics for imbalanced datasets
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    
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
        'balanced_accuracy': balanced_acc,
        'mcc': mcc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def get_model_size(model):
    """Calculate the size of a trained model in MB or KB."""
    import sys
    import pickle
    
    try:
        # Serialize the model to get its size
        model_bytes = pickle.dumps(model)
        size_bytes = len(model_bytes)
        
        # Convert to appropriate unit
        if size_bytes >= 1024 * 1024:  # >= 1 MB
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f} MB"
        else:
            size_kb = size_bytes / 1024
            return f"{size_kb:.2f} KB"
            
    except Exception as e:
        return f"Error calculating size: {str(e)}"

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
    # Create main results plot (metrics, ROC, PR curves)
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Credit Card Fraud Detection - Model Comparison', fontsize=16)
    
    # 1. Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auc_pr', 'balanced_accuracy', 'mcc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR', 'Balanced Acc', 'MCC']
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (model_name, result) in enumerate(results.items()):
        values = [result[metric] for metric in metrics]
        axes1[0].bar(x + i * width, values, width, label=model_name, alpha=0.8)
    
    axes1[0].set_xlabel('Metrics')
    axes1[0].set_ylabel('Score')
    axes1[0].set_title('Model Performance Comparison')
    axes1[0].set_xticks(x + width * 1.5)
    axes1[0].set_xticklabels(metric_names, rotation=45)
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # 2. ROC Curves
    for model_name, result in results.items():
        axes1[1].plot(result['fpr'], result['tpr'], label=f"{model_name} (AUC={result['auc']:.3f})")
    
    axes1[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes1[1].set_xlabel('False Positive Rate')
    axes1[1].set_ylabel('True Positive Rate')
    axes1[1].set_title('ROC Curves')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curves
    for model_name, result in results.items():
        axes1[2].plot(result['recall_curve'], result['precision_curve'], 
                       label=f"{model_name} (AUC-PR={result['auc_pr']:.3f})")
    
    axes1[2].set_xlabel('Recall')
    axes1[2].set_ylabel('Precision')
    axes1[2].set_title('Precision-Recall Curves')
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate confusion matrices plot
    n_models = len(results)
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('Credit Card Fraud Detection - Confusion Matrices', fontsize=16)
    
    axes2 = axes2.flatten()
    
    for i, (model_name, result) in enumerate(results.items()):
        if i >= 4:  # Safety check
            break
            
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[i])
        axes2[i].set_title(f'{model_name} - Confusion Matrix')
        axes2[i].set_xlabel('Predicted')
        axes2[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
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

def visualize_neural_network(nn_model, feature_names):
    """Visualize the Neural Network architecture and learned weights."""
    print("\n" + "="*80)
    print("NEURAL NETWORK MODEL ANALYSIS")
    print("="*80)
    
    # Get model architecture details
    n_features = len(feature_names)
    hidden_layers = nn_model.hidden_layer_sizes
    n_outputs = 1  # Binary classification
    
    # Calculate total parameters manually since n_parameters_ might not be available
    total_params = 0
    if hasattr(nn_model, 'coefs_') and hasattr(nn_model, 'intercepts_'):
        for i, coef in enumerate(nn_model.coefs_):
            total_params += coef.size
        for intercept in nn_model.intercepts_:
            total_params += intercept.size
    
    print(f"Architecture: {n_features} → {hidden_layers} → {n_outputs}")
    print(f"Total parameters: {total_params:,}")
    print(f"Activation function: {nn_model.activation}")
    print(f"Learning rate: {getattr(nn_model, 'learning_rate_init', 'Not available')}")
    print(f"Max iterations: {nn_model.max_iter}")
    print(f"Convergence: {'Yes' if hasattr(nn_model, 'n_iter_') and nn_model.n_iter_ < nn_model.max_iter else 'No'}")
    print(f"Iterations to converge: {getattr(nn_model, 'n_iter_', 'Not available')}")
    
    # Create network architecture visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Network Architecture Diagram
    ax1.set_title('Neural Network Architecture', fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Draw layers
    layer_positions = [0, 1, 2, 3]
    layer_sizes = [n_features] + list(hidden_layers) + [n_outputs]
    layer_names = ['Input\n(30 features)'] + [f'Hidden {i+1}\n({size} neurons)' for i, size in enumerate(hidden_layers)] + ['Output\n(1 neuron)']
    
    for i, (pos, size, name) in enumerate(zip(layer_positions, layer_sizes, layer_names)):
        # Draw neurons as circles
        for j in range(size):
            y_pos = 9 - (j * 8 / max(size, 1))
            circle = plt.Circle((pos, y_pos), 0.3, fill=True, color='lightblue', edgecolor='black', linewidth=1)
            ax1.add_patch(circle)
        
        # Add layer label
        ax1.text(pos, -0.5, name, ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Draw connections (simplified)
    for i in range(len(layer_positions) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                y1 = 9 - (j * 8 / max(layer_sizes[i], 1))
                y2 = 9 - (k * 8 / max(layer_sizes[i + 1], 1))
                ax1.plot([layer_positions[i] + 0.3, layer_positions[i + 1] - 0.3], [y1, y2], 
                        'k-', alpha=0.1, linewidth=0.5)
    
    # 2. Weight Distribution Analysis
    ax2.set_title('Weight Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Get weight matrices
    coefs = nn_model.coefs_
    intercepts = nn_model.intercepts_
    
    # Flatten all weights for distribution analysis
    all_weights = []
    for coef in coefs:
        all_weights.extend(coef.flatten())
    
    # Plot weight distribution
    ax2.hist(all_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Zero')
    ax2.axvline(x=np.mean(all_weights), color='green', linestyle='--', alpha=0.8, 
                label=f'Mean: {np.mean(all_weights):.4f}')
    ax2.axvline(x=np.std(all_weights), color='orange', linestyle='--', alpha=0.8, 
                label=f'Std: {np.std(all_weights):.4f}')
    
    ax2.set_xlabel('Weight Values')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neural_network_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed weight statistics
    print(f"\nWeight Statistics:")
    print(f"  - Total weights: {len(all_weights):,}")
    print(f"  - Mean weight: {np.mean(all_weights):.6f}")
    print(f"  - Std weight: {np.std(all_weights):.6f}")
    print(f"  - Min weight: {np.min(all_weights):.6f}")
    print(f"  - Max weight: {np.max(all_weights):.6f}")
    print(f"  - Zero weights: {np.sum(np.abs(all_weights) < 1e-10):,} ({np.sum(np.abs(all_weights) < 1e-10)/len(all_weights)*100:.2f}%)")
    
    # Layer-by-layer analysis
    print(f"\nLayer-by-Layer Analysis:")
    for i, (coef, intercept) in enumerate(zip(coefs, intercepts)):
        layer_name = "Input → Hidden 1" if i == 0 else f"Hidden {i} → Hidden {i+1}" if i < len(coefs)-1 else "Hidden → Output"
        print(f"  {layer_name}:")
        print(f"    - Weights shape: {coef.shape}")
        print(f"    - Bias shape: {intercept.shape}")
        print(f"    - Weight range: [{coef.min():.4f}, {coef.max():.4f}]")
        print(f"    - Bias range: [{intercept.min():.4f}, {intercept.max():.4f}]")
    
    return fig

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
            'AUC-PR': f"{result['auc_pr']:.4f}",
            'Balanced Acc': f"{result['balanced_accuracy']:.4f}",
            'MCC': f"{result['mcc']:.4f}"
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
        print(f"Balanced Accuracy: {result['balanced_accuracy']:.4f}")
        print(f"Matthews Correlation: {result['mcc']:.4f}")
        print(f"Specificity: {result['specificity']:.4f}")
        print(f"Sensitivity: {result['sensitivity']:.4f}")
        
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
    # Set use_class_weights=False to disable class weight handling
    models = train_models(X_train_scaled, X_val_scaled, y_train, y_val, use_class_weights=False)
    
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
    
    # Print detailed model size analysis
    print("\n" + "="*80)
    print("DETAILED MODEL SIZE ANALYSIS")
    print("="*80)
    
    for model_name, model in models.items():
        model_size = get_model_size(model)
        print(f"\n{model_name}:")
        print(f"  - Size: {model_size}")
        
        # Additional model-specific information
        if hasattr(model, 'n_estimators'):
            print(f"  - Number of estimators: {model.n_estimators}")
        if hasattr(model, 'hidden_layer_sizes'):
            print(f"  - Hidden layers: {model.hidden_layer_sizes}")
        if hasattr(model, 'max_depth'):
            print(f"  - Max depth: {model.max_depth}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_results(results)
    plot_feature_importance(feature_importance_dict)
    
    # Visualize Neural Network specifically
    if 'Neural Network' in models:
        print("\nGenerating Neural Network visualization...")
        nn_model = models['Neural Network']
        visualize_neural_network(nn_model, X.columns)
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    summary_data = []
    for model_name, result in results.items():
        # Get model size for this model
        model = models[model_name]
        model_size = get_model_size(model)
        
        summary_data.append({
            'Model': model_name,
            'Model_Size': model_size,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1'],
            'AUC_ROC': result['auc'],
            'AUC_PR': result['auc_pr'],
            'Balanced_Accuracy': result['balanced_accuracy'],
            'MCC': result['mcc'],
            'Specificity': result['specificity'],
            'Sensitivity': result['sensitivity']
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
    print("- confusion_matrices.png")
    print("- feature_importance.png")
    print("- neural_network_analysis.png")
    print("- Individual feature importance CSV files")

if __name__ == "__main__":
    main() 