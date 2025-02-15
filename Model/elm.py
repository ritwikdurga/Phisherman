import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

# Create directory for pickle files
PICKLE_DIR = "pickle_files"
os.makedirs(PICKLE_DIR, exist_ok=True)

def save_model(model, filename):
    pickle.dump(model, open(os.path.join(PICKLE_DIR, filename), "wb"))

def load_model(filename):
    return pickle.load(open(os.path.join(PICKLE_DIR, filename), "rb"))

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
df = pd.read_csv(url, skiprows=96, header=None)

# Preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = np.where(y == -1, 0, 1)  # Convert labels to 0/1

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler_path = os.path.join(PICKLE_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = load_model("scaler.pkl")
else:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    save_model(scaler, "scaler.pkl")

# ELM Classifier
class ExtremeLearningMachine(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_units=1000, activation='sigmoid'):
        self.hidden_units = hidden_units
        self.activation = activation

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def fit(self, X, y):
        limit = np.sqrt(6 / (X.shape[1] + self.hidden_units))
        self.input_weights = np.random.uniform(-limit, limit, (X.shape[1], self.hidden_units))
        self.biases = np.random.uniform(-limit, limit, (1, self.hidden_units))

        H = np.dot(X, self.input_weights) + self.biases
        H = self._relu(H) if self.activation == 'relu' else self._sigmoid(H)

        I = np.eye(self.hidden_units)
        self.output_weights = np.linalg.inv(H.T @ H + 1e-6 * I) @ H.T @ y

    def predict(self, X):
        H = np.dot(X, self.input_weights) + self.biases
        H = self._relu(H) if self.activation == 'relu' else self._sigmoid(H)
        return np.dot(H, self.output_weights).round().astype(int)

# Load or train function
def load_or_train(model_name, model_class, *args, **kwargs):
    path = os.path.join(PICKLE_DIR, model_name)
    if os.path.exists(path):
        print(f"Loading {model_name}...")
        return load_model(model_name)
    else:
        print(f"Training {model_name}...")
        model = model_class(*args, **kwargs)
        model.fit(X_train, y_train)
        save_model(model, model_name)
        return model

# Grid Search for Best ELM
def grid_search_elm():
    param_grid = {'hidden_units': [500, 1000, 1500], 'activation': ['sigmoid', 'relu']}
    grid = GridSearchCV(ExtremeLearningMachine(), param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid.best_estimator_

best_elm_path = os.path.join(PICKLE_DIR, "best_elm.pkl")
if os.path.exists(best_elm_path):
    best_elm = load_model("best_elm.pkl")
else:
    best_elm = grid_search_elm()
    save_model(best_elm, "best_elm.pkl")

# Ensemble ELM
class EnsembleELM:
    def __init__(self, hidden_units_list=[800, 1000, 1200]):
        self.elm_models = [ExtremeLearningMachine(hidden_units=hu, activation='relu') for hu in hidden_units_list]

    def fit(self, X, y):
        for model in self.elm_models:
            model.fit(X, y)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.elm_models])
        return np.round(np.mean(predictions, axis=0)).astype(int)

ensemble_path = os.path.join(PICKLE_DIR, "ensemble_elm.pkl")
if os.path.exists(ensemble_path):
    ensemble = load_model("ensemble_elm.pkl")
else:
    ensemble = EnsembleELM()
    ensemble.fit(X_train, y_train)
    save_model(ensemble, "ensemble_elm.pkl")

# Load or train classifiers
elm_model = load_or_train("elm_model.pkl", ExtremeLearningMachine, hidden_units=1000)
svm_model = load_or_train("svm_model.pkl", SVC, kernel='rbf')
nb_model = load_or_train("nb_model.pkl", GaussianNB)

# Evaluate models
models = {
    "ELM": elm_model,
    "Best ELM (Grid Search)": best_elm,
    "Ensemble ELM": ensemble,
    "SVM": svm_model,
    "Naïve Bayes": nb_model
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {'accuracy': accuracy, 'confusion_matrix': cm, 'report': report}

# Best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
print(f"🏆 Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
accuracies = [results[model]['accuracy'] for model in results]
sns.barplot(x=list(results.keys()), y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
plt.subplot(1, 2, 2)
sns.heatmap(results[best_model_name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.title(f'{best_model_name} Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

for model_name, result in results.items():
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:")
    print(result['report'])
