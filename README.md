## Credit Card Fraud Detection – Model Comparison

This project compares multiple machine learning models for credit card fraud detection on the widely used imbalanced dataset of European card transactions. It trains and evaluates:

- Random Forest
- Gradient Boosting (GBDT)
- Logistic Regression
- Feedforward Neural Network (MLP)

The script produces comprehensive metrics, plots (ROC and Precision–Recall), confusion matrices, and feature importance analyses where applicable.

### Dataset
- Source: "Credit Card Fraud Detection" (Kaggle; originally from Worldline and ULB). The dataset contains PCA-transformed features `V1`–`V28` plus `Time` and `Amount`, with target `Class` (1 = fraud).
- Download the dataset and place `creditcard.csv` in the project root.
  - Dataset link: `https://www.kaggle.com/mlg-ulb/creditcardfraud`

### Project Structure
- `credit_card_fraud_analysis.py`: Main training/evaluation script
- `analysis_summary.md`: Human-readable summary of results and insights
- `requirements.txt`: Python dependencies
- `venv/`: Optional local virtual environment directory

### Quick Start
1) Create and activate a virtual environment (recommended)
   - Windows PowerShell:
     ```bash
     python -m venv venv
     .\\venv\\Scripts\\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Place the dataset at the project root as `creditcard.csv`

4) Run the analysis
```bash
python credit_card_fraud_analysis.py
```

### What the Script Does
The pipeline in `credit_card_fraud_analysis.py`:
- Loads `creditcard.csv`
- Splits data into train/validation/test with stratification
- Scales features (StandardScaler) and trains all models on scaled features for consistency
- **Handles class imbalance** using class weights and sample weights for better fraud detection
- Evaluates models using comprehensive metrics including balanced accuracy and Matthews correlation
- Saves a CSV summary and plots for model comparison and feature importance

### Outputs
After a successful run, the following files are generated in the project root:
- `model_comparison_results.csv`: Tabular summary of metrics per model
- `model_comparison_results.png`: Bar charts, ROC curves, and Precision-Recall curves
- `confusion_matrices.png`: Confusion matrices for all four models
- `feature_importance.png`: Top features for models that expose importances/coefficients
- `<ModelName>_feature_importance.csv`: Per-model feature importance where available

### Class Imbalance Handling
The dataset has a severe class imbalance (577:1 legitimate:fraud ratio). The script addresses this by:
- **Random Forest & Logistic Regression**: Using `class_weight='balanced'` parameter
- **GBDT & Neural Network**: Using sample weights calculated from class distribution
- **Evaluation**: Including balanced accuracy, Matthews correlation coefficient, and specificity metrics

### Configuration Tips
- You can adjust model hyperparameters in `train_models` within `credit_card_fraud_analysis.py`.
- To emphasize recall (catch more fraud), consider threshold tuning using the predicted probabilities stored in the results.
- Class weights are automatically calculated based on training data distribution.