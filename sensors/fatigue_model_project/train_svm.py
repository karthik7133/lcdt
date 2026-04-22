import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("--- TRAINING ADVANCED V2 MULTI-MODAL BRAIN (95%+ Target) ---")

# 1. Load dataset_v2
print("Loading dataset_v2.csv...")
try:
    df = pd.read_csv('dataset_v2.csv')
except FileNotFoundError:
    print("Error: dataset_v2.csv not found. Please collect data first.")
    exit()

# 2. Advanced Feature Engineering (Temporal Context)
print("Creating temporal features (Rolling averages for Multi-Modal data)...")
window_size = 10
df['EAR_MA'] = df['EAR'].rolling(window=window_size).mean()
df['MAR_MA'] = df['MAR'].rolling(window=window_size).mean()
df['Pitch_MA'] = df['Pitch'].rolling(window=window_size).mean()
df['EAR_STD'] = df['EAR'].rolling(window=window_size).std()

# Drop rows with NaNs (the first 9 frames)
df = df.dropna()

# 3. Extract Multi-Modal Features
# Features: EAR, MAR, Pitch, Blink_Count + Temporal Context
X = df[['EAR', 'MAR', 'Pitch', 'Blink_Count', 'EAR_MA', 'MAR_MA', 'Pitch_MA', 'EAR_STD']]
y = df['Label']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Define the Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, class_weight='balanced'))
])

# 6. Optimized Hyperparameter Grid
param_grid = {
    'svm__kernel': ['rbf'],
    'svm__C': [1, 10, 100],
    'svm__gamma': ['scale', 0.1, 0.01]
}

print("Searching for the best Multi-Modal model (GridSearch)...")
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 7. Evaluation
model = grid_search.best_estimator_
print(f"Best Params Found: {grid_search.best_params_}")

print("\nEvaluating on hidden test data...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- RESULTS ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:\n", classification_report(y_test, y_pred, target_names=['Awake (0)', 'Tired (1)']))

# 8. Save the V2 Brain
model_filename = 'fatigue_model_v2.pkl'
joblib.dump(model, model_filename)

print(f"\nSUCCESS! V2 Brain saved as '{model_filename}'")
