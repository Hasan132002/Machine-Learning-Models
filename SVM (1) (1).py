import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\Hasan Raza\Desktop\ML\ML PROJECTS\Cancer_Data.csv")


# Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Fix NaN values
X = SimpleImputer(strategy='mean').fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# FAST SVM
model = LinearSVC(max_iter=5000)

# Train
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predictions
predictions = model.predict(X_test)
print("Predictions:", predictions)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Accuracy in percentage
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


