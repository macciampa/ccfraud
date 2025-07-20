import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred, digits=4))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Save the trained model
joblib.dump(model, 'logistic_regression_model.joblib')
print('Model saved as logistic_regression_model.joblib') 