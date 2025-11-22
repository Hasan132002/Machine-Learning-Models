import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv(r"C:\Users\Warda Ghias\Desktop\ML\ML PROJECTS\Cancer_Data.csv")

# Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Fix NaN values
X = SimpleImputer(strategy='mean').fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

# Train
rf.fit(X_train, y_train)

# Predict
predictions = rf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Predictions
print("\nPredictions:", predictions)
