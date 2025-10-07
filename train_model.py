import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder

# Load dataset and clean column names
df = pd.read_csv("admission_data_universities.csv")
df.columns = df.columns.str.strip()

# Label encode categorical features
le_course = LabelEncoder()
df['Preferred Course'] = le_course.fit_transform(df['Preferred Course'].astype(str))

le_country = LabelEncoder()
df['Preferred Country'] = le_country.fit_transform(df['Preferred Country'].astype(str))

# Create binary target
df["Admit"] = df["Chance of Admit"].apply(lambda x: 1 if x >= 0.75 else 0)

# Define features including preferred course and country
features = ['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR', 'Research',
            'Preferred Course', 'Preferred Country']
X = df[features]
y = df["Admit"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save model and encoders for use during prediction
joblib.dump(model, "admission_model.pkl")
joblib.dump(le_course, "label_encoder_course.pkl")
joblib.dump(le_country, "label_encoder_country.pkl")
print("Model and label encoders saved.")

# Plot feature importance (optional)
plot_importance(model, importance_type='gain', title='Feature Importance')
plt.tight_layout()
plt.show()
