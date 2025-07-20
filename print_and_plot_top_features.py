import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the dataset and model
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
model = joblib.load('logistic_regression_model.joblib')

# Get feature importances
feature_importance = np.abs(model.coef_[0])
feature_names = X.columns
important_indices = np.argsort(feature_importance)[-10:][::-1]
top_features = feature_names[important_indices]
top_importances = feature_importance[important_indices]

# Print the 10 most important features
print('Top 10 most important features:')
for name, importance in zip(top_features, top_importances):
    print(f'{name}: {importance:.4f}')

# Show histograms for the top 3 features
plt.figure(figsize=(15, 4))
for i, feature in enumerate(top_features[:3]):
    plt.subplot(1, 3, i+1)
    plt.hist(X[feature], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show() 