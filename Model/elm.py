import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
df = pd.read_csv(url, skiprows=96, header=None)

# Preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = np.where(y == -1, 0, 1)  # Convert labels to 0/1

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Enhanced ELM Classifier with multiple activation functions and regularization
class EnhancedELM(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_units=1000, activation='relu', alpha=1e-4, weight_init='he'):
        self.hidden_units = hidden_units
        self.activation = activation
        self.alpha = alpha  # Regularization parameter
        self.weight_init = weight_init

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)

    def _elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def _swish(self, x, beta=1.0):
        return x * self._sigmoid(beta * x)

    def _initialize_weights(self, input_size, hidden_size):
        if self.weight_init == 'uniform':
            limit = np.sqrt(6 / (input_size + hidden_size))
            return np.random.uniform(-limit, limit, (input_size, hidden_size))
        elif self.weight_init == 'he':  # Good for ReLU
            return np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        elif self.weight_init == 'xavier':  # Good for sigmoid/tanh
            return np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        else:
            raise ValueError(f"Unknown weight initialization: {self.weight_init}")

    def _apply_activation(self, x):
        if self.activation == 'sigmoid':
            return self._sigmoid(x)
        elif self.activation == 'relu':
            return self._relu(x)
        elif self.activation == 'leaky_relu':
            return self._leaky_relu(x)
        elif self.activation == 'elu':
            return self._elu(x)
        elif self.activation == 'swish':
            return self._swish(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def fit(self, X, y):
        # Initialize weights
        self.input_weights = self._initialize_weights(X.shape[1], self.hidden_units)
        self.biases = np.random.uniform(-0.1, 0.1, (1, self.hidden_units))

        # Calculate hidden layer output
        H = np.dot(X, self.input_weights) + self.biases
        H = self._apply_activation(H)

        # Regularized pseudo-inverse calculation with tunable alpha
        I = np.eye(self.hidden_units)
        self.output_weights = np.linalg.inv(H.T @ H + self.alpha * I) @ H.T @ y

        return self

    def predict(self, X):
        # Calculate hidden layer output
        H = np.dot(X, self.input_weights) + self.biases
        H = self._apply_activation(H)

        return np.dot(H, self.output_weights).round().astype(int)

    def predict_proba(self, X):
        # Calculate hidden layer output
        H = np.dot(X, self.input_weights) + self.biases
        H = self._apply_activation(H)

        # Get raw outputs
        raw_output = np.dot(H, self.output_weights)

        # Convert to probabilities using sigmoid
        probs = self._sigmoid(raw_output)
        return np.column_stack((1 - probs, probs))


# Enhanced Ensemble ELM with weighted averaging
class EnhancedEnsembleELM:
    def __init__(self, use_weights=True):
        # Create a diverse set of models with different configurations
        self.models = [
            EnhancedELM(hidden_units=800, activation='relu', alpha=1e-4, weight_init='he'),
            EnhancedELM(hidden_units=1000, activation='leaky_relu', alpha=1e-3, weight_init='he'),
            EnhancedELM(hidden_units=1200, activation='elu', alpha=1e-4, weight_init='he'),
            EnhancedELM(hidden_units=1500, activation='swish', alpha=1e-5, weight_init='xavier'),
            EnhancedELM(hidden_units=2000, activation='relu', alpha=1e-3, weight_init='he')
        ]

        self.use_weights = use_weights
        self.weights = None

    def fit(self, X, y):
        # Split training data for validation to determine model weights
        if self.use_weights:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Train each model on the training portion
            for model in self.models:
                model.fit(X_train, y_train)

            # Calculate weights based on validation performance
            self.weights = []
            for model in self.models:
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                self.weights.append(val_acc)

            # Normalize weights
            self.weights = np.array(self.weights) / sum(self.weights)

            # Retrain on full dataset
            for model in self.models:
                model.fit(X, y)
        else:
            # Train each model on the full dataset
            for model in self.models:
                model.fit(X, y)

            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)

    def predict(self, X):
        # Get predictions from each model
        predictions = np.array([model.predict(X) for model in self.models])

        # Apply weighted average
        weighted_sum = np.zeros(X.shape[0])
        for i, model_pred in enumerate(predictions):
            weighted_sum += self.weights[i] * model_pred

        return np.round(weighted_sum).astype(int)


# Bagging Ensemble ELM
class BaggingEnsembleELM:
    def __init__(self, n_estimators=10, sample_ratio=0.8):
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.models = [EnhancedELM(hidden_units=1000, activation='relu', alpha=1e-4) for _ in range(n_estimators)]

    def fit(self, X, y):
        n_samples = int(X.shape[0] * self.sample_ratio)

        for i, model in enumerate(self.models):
            # Bootstrap sampling
            indices = np.random.choice(X.shape[0], n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            # Train model on bootstrap sample
            model.fit(X_sample, y_sample)

        return self

    def predict(self, X):
        # Get predictions from each model
        predictions = np.array([model.predict(X) for model in self.models])

        # Majority voting
        return np.round(np.mean(predictions, axis=0)).astype(int)


# Stacking Ensemble
class StackingEnsemble:
    def __init__(self):
        # Base models
        self.base_models = [
            EnhancedELM(hidden_units=800, activation='relu', alpha=1e-4),
            EnhancedELM(hidden_units=1200, activation='leaky_relu', alpha=1e-3),
            EnhancedELM(hidden_units=1500, activation='elu', alpha=1e-4),
            EnhancedELM(hidden_units=1000, activation='swish', alpha=1e-5)
        ]

        # Meta-learner
        self.meta_learner = EnhancedELM(hidden_units=100, activation='sigmoid', alpha=1e-4)

    def fit(self, X, y):
        # Split training data for meta-learner
        X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train base models
        for model in self.base_models:
            model.fit(X_train_base, y_train_base)

        # Generate meta-features
        meta_features = np.column_stack([
            model.predict_proba(X_train_meta)[:, 1] for model in self.base_models
        ])

        # Train meta-learner
        self.meta_learner.fit(meta_features, y_train_meta)

        # Retrain base models on full training data
        for model in self.base_models:
            model.fit(X, y)

        return self

    def predict(self, X):
        # Generate meta-features
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])

        # Predict with meta-learner
        return self.meta_learner.predict(meta_features)


# Train and evaluate models
print("Training basic ELM model...")
elm_model = EnhancedELM(hidden_units=1000, activation='relu')
elm_model.fit(X_train, y_train)

print("Training enhanced ensemble ELM...")
ensemble = EnhancedEnsembleELM()
ensemble.fit(X_train, y_train)

print("Training bagging ensemble ELM...")
bagging = BaggingEnsembleELM(n_estimators=10)
bagging.fit(X_train, y_train)

print("Training stacking ensemble...")
stacking = StackingEnsemble()
stacking.fit(X_train, y_train)

# Evaluate models
models = {
    "Basic ELM": elm_model,
    "Enhanced Ensemble ELM": ensemble,
    "Bagging Ensemble ELM": bagging,
    "Stacking Ensemble": stacking
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    results[name] = {'accuracy': accuracy, 'confusion_matrix': cm, 'report': report}

# Find the best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']

print(f"🏆 Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Visualization
plt.figure(figsize=(15, 5))

# Accuracy comparison
plt.subplot(1, 2, 1)
accuracies = [results[model]['accuracy'] for model in results]
sns.barplot(x=list(results.keys()), y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
plt.xticks(rotation=45, ha='right')

# Confusion matrix for Best Model
plt.subplot(1, 2, 2)
sns.heatmap(results[best_model_name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.title(f'{best_model_name} Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Print results
for model_name, result in results.items():
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:")
    print(result['report'])



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# import os
# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.base import BaseEstimator, ClassifierMixin

# # Create directory for pickle files
# PICKLE_DIR = "pickle_files"
# os.makedirs(PICKLE_DIR, exist_ok=True)

# def save_model(model, filename):
#     pickle.dump(model, open(os.path.join(PICKLE_DIR, filename), "wb"))

# def load_model(filename):
#     return pickle.load(open(os.path.join(PICKLE_DIR, filename), "rb"))

# # Load dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
# df = pd.read_csv(url, skiprows=96, header=None)

# # Preprocessing
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values
# y = np.where(y == -1, 0, 1)  # Convert labels to 0/1

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Feature scaling
# scaler_path = os.path.join(PICKLE_DIR, "scaler.pkl")
# if os.path.exists(scaler_path):
#     scaler = load_model("scaler.pkl")
# else:
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     save_model(scaler, "scaler.pkl")

# # ELM Classifier
# class ExtremeLearningMachine(BaseEstimator, ClassifierMixin):
#     def __init__(self, hidden_units=1000, activation='sigmoid'):
#         self.hidden_units = hidden_units
#         self.activation = activation

#     def _sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def _relu(self, x):
#         return np.maximum(0, x)

#     def fit(self, X, y):
#         limit = np.sqrt(6 / (X.shape[1] + self.hidden_units))
#         self.input_weights = np.random.uniform(-limit, limit, (X.shape[1], self.hidden_units))
#         self.biases = np.random.uniform(-limit, limit, (1, self.hidden_units))

#         H = np.dot(X, self.input_weights) + self.biases
#         H = self._relu(H) if self.activation == 'relu' else self._sigmoid(H)

#         I = np.eye(self.hidden_units)
#         self.output_weights = np.linalg.inv(H.T @ H + 1e-6 * I) @ H.T @ y

#     def predict(self, X):
#         H = np.dot(X, self.input_weights) + self.biases
#         H = self._relu(H) if self.activation == 'relu' else self._sigmoid(H)
#         return np.dot(H, self.output_weights).round().astype(int)

# # Load or train function
# def load_or_train(model_name, model_class, *args, **kwargs):
#     path = os.path.join(PICKLE_DIR, model_name)
#     if os.path.exists(path):
#         print(f"Loading {model_name}...")
#         return load_model(model_name)
#     else:
#         print(f"Training {model_name}...")
#         model = model_class(*args, **kwargs)
#         model.fit(X_train, y_train)
#         save_model(model, model_name)
#         return model

# # Grid Search for Best ELM
# def grid_search_elm():
#     param_grid = {'hidden_units': [500, 1000, 1500], 'activation': ['sigmoid', 'relu']}
#     grid = GridSearchCV(ExtremeLearningMachine(), param_grid, cv=3, scoring='accuracy')
#     grid.fit(X_train, y_train)
#     return grid.best_estimator_

# best_elm_path = os.path.join(PICKLE_DIR, "best_elm.pkl")
# if os.path.exists(best_elm_path):
#     best_elm = load_model("best_elm.pkl")
# else:
#     best_elm = grid_search_elm()
#     save_model(best_elm, "best_elm.pkl")

# # Ensemble ELM
# class EnsembleELM:
#     def __init__(self, hidden_units_list=[800, 1000, 1200]):
#         self.elm_models = [ExtremeLearningMachine(hidden_units=hu, activation='relu') for hu in hidden_units_list]

#     def fit(self, X, y):
#         for model in self.elm_models:
#             model.fit(X, y)

#     def predict(self, X):
#         predictions = np.array([model.predict(X) for model in self.elm_models])
#         return np.round(np.mean(predictions, axis=0)).astype(int)

# ensemble_path = os.path.join(PICKLE_DIR, "ensemble_elm.pkl")
# if os.path.exists(ensemble_path):
#     ensemble = load_model("ensemble_elm.pkl")
# else:
#     ensemble = EnsembleELM()
#     ensemble.fit(X_train, y_train)
#     save_model(ensemble, "ensemble_elm.pkl")

# # Load or train classifiers
# elm_model = load_or_train("elm_model.pkl", ExtremeLearningMachine, hidden_units=1000)
# svm_model = load_or_train("svm_model.pkl", SVC, kernel='rbf')
# nb_model = load_or_train("nb_model.pkl", GaussianNB)

# # Evaluate models
# models = {
#     "ELM": elm_model,
#     "Best ELM (Grid Search)": best_elm,
#     "Ensemble ELM": ensemble,
#     "SVM": svm_model,
#     "Naïve Bayes": nb_model
# }

# results = {}
# for name, model in models.items():
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     results[name] = {'accuracy': accuracy, 'confusion_matrix': cm, 'report': report}

# # Best model
# best_model_name = max(results, key=lambda k: results[k]['accuracy'])
# best_accuracy = results[best_model_name]['accuracy']
# print(f"🏆 Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# # Visualization
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# accuracies = [results[model]['accuracy'] for model in results]
# sns.barplot(x=list(results.keys()), y=accuracies)
# plt.title('Model Accuracy Comparison')
# plt.ylabel('Accuracy')
# plt.ylim(0.8, 1.0)
# plt.subplot(1, 2, 2)
# sns.heatmap(results[best_model_name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
# plt.title(f'{best_model_name} Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.tight_layout()
# plt.show()

# for model_name, result in results.items():
#     print(f"\n{model_name} Results:")
#     print(f"Accuracy: {result['accuracy']:.4f}")
#     print("Classification Report:")
#     print(result['report'])
