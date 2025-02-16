import numpy as np
import pandas as pd
import joblib
from train_final import DGR_ELM  

data = joblib.load("dgr_elm_model.pkl")  
elm = data["model"]
X_mean = data["mean"]
X_std = data["std"]

input_data = pd.read_csv("new_fault_data.csv", header=None)

X_new = input_data.values


if X_new.ndim == 1:
    X_new = X_new.reshape(1, -1)  

if X_new.shape[1] != X_mean.shape[0]:
    raise ValueError(f"Mismatch: Expected {X_mean.shape[0]} features, but got {X_new.shape[1]}")

X_std[X_std == 0] = 1  
X_new = (X_new - X_mean) / X_std

predictions = elm.predict(X_new).astype(int)

fault_mapping = {
    "0000": "No Fault",
    "1001": "LG Fault (Phase A-Ground)",
    "0011": "LL Fault (Phase A-B)",
    "1011": "LLG Fault (Phase A-B-Ground)",
    "0111": "LLL Fault (Three-Phase)",
    "1111": "LLLG Fault (Three-Phase + Ground)"
}

fault_types = []
for pred in predictions:
    fault_label = "".join(map(str, pred))  
    fault_types.append(fault_mapping.get(fault_label, "Unknown Fault"))

output_df = pd.DataFrame({"Fault Type": fault_types})
output_df.to_csv("fault_predictions.csv", index=False)

