import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Load dataset
df = pd.read_csv("admission_data_universities.csv")

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Print to verify column names (optional)
print("Columns:", df.columns.tolist())

# Define features and target
features = ["GRE Score", "TOEFL Score", "CGPA", "SOP", "LOR", "Research"]
target = "Chance of Admit"

# Create binary target: 1 if high chance, 0 if low
df["Admit"] = df[target].apply(lambda x: 1 if x >= 0.75 else 0)

# Prepare input and output
X = df[features]
y = df["Admit"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save trained model
joblib.dump(model, "admission_model.pkl")
print("Model saved as 'admission_model.pkl'")



plot_importance(model, importance_type='gain', title='Feature Importance')
plt.tight_layout()
plt.show()

