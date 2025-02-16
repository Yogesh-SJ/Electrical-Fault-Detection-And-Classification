import numpy as np
import pandas as pd
from scipy.linalg import pinv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class DGR_ELM:
    def __init__(self, num_hidden_neurons=500, lambda2=0.1):
        self.num_hidden_neurons = num_hidden_neurons
        self.lambda2 = lambda2  

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y):
        num_samples, num_features = X.shape
        num_classes = y.shape[1]  

        self.input_weights = np.random.randn(self.num_hidden_neurons, num_features)
        self.biases = np.random.randn(self.num_hidden_neurons)

        H = self._sigmoid(np.dot(X, self.input_weights.T) + self.biases)

        identity_matrix = np.identity(self.num_hidden_neurons)
        self.output_weights = pinv(H.T @ H + self.lambda2 * identity_matrix) @ H.T @ y

    def predict(self, X):
        H = self._sigmoid(np.dot(X, self.input_weights.T) + self.biases)
        Y_pred = np.dot(H, self.output_weights)
        Y_pred = (Y_pred > 0.5).astype(int)  
        return Y_pred

if __name__ == "__main__":
    df = pd.read_csv("classData.csv")

    X = df.iloc[:, 4:].values  
    y = df.iloc[:, 0:4].values  

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  

    X = (X - X_mean) / X_std
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    elm = DGR_ELM(num_hidden_neurons=500)
    elm.train(X_train, y_train)

    y_pred = elm.predict(X_test)

    y_pred = y_pred.astype(int)
    y_test = y_test.astype(int)

    print("y_test shape:", y_test.shape, "y_pred shape:", y_pred.shape)
    print("Sample y_test:", y_test[:5])
    print("Sample y_pred:", y_pred[:5])

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    joblib.dump({"model": elm, "mean": X_mean, "std": X_std}, "dgr_elm_model.pkl")
    print("Model saved as dgr_elm_model.pkl")