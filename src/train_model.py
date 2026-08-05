import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("dataset/diabetes.csv")

print("First 5 Rows:")
print(data.head())

# -----------------------------
# Create Models Folder
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()

data["gender"] = gender_encoder.fit_transform(data["gender"])
data["smoking_history"] = smoking_encoder.fit_transform(data["smoking_history"])

# Save Encoders
joblib.dump(gender_encoder, "models/gender_encoder.pkl")
joblib.dump(smoking_encoder, "models/smoking_encoder.pkl")

# -----------------------------
# Fill Missing Values
# -----------------------------
imputer = SimpleImputer(strategy="median")
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# -----------------------------
# Features and Target
# -----------------------------
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "models/scaler.pkl")

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_accuracy = 0
best_name = ""

print("\nTraining Models...\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name}: {accuracy*100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print("\n==============================")
print("Best Model :", best_name)
print(f"Accuracy   : {best_accuracy*100:.2f}%")
print("==============================")

# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(best_model, "models/diabetes_model.pkl")

print("\nModel Saved Successfully!")