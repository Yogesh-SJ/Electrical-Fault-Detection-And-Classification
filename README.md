# Fault Classification Using DGR-ELM

## Introduction

This project focuses on classifying different types of electrical faults using a deep learning model based on the **DGR-ELM (Distributed Generalized Regularized Extreme Learning Machine) algorithm**. The dataset consists of electrical parameters such as current and voltage values to identify various fault types. The model is trained using `train_final.py`, and predictions are made using `predict1.py`.

## Features

- **Fault Classification:** Identifies different fault types based on input current and voltage values.
- **Deep Learning Model:** Uses DGR-ELM for accurate and efficient classification.
- **Dataset Handling:** Stores training and testing data in `classData.csv`.
- **Prediction Script:** `predict1.py` allows real-time fault prediction.
- **Training Script:** `train_final.py` trains the fault classification model.

## Installation

To set up the project on your local machine, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Creating Virtual Environments

To create an isolated environment for running the project:

```bash
python -m venv fault_classification_env
source fault_classification_env/bin/activate  # On Linux/macOS
fault_classification_env\Scripts\activate    # On Windows
pip install -r requirements.txt
```

## Dataset

The dataset (`classData.csv`) contains fault classification data with the following structure:

### Inputs (Features):

- **Ia, Ib, Ic** - Phase currents
- **Va, Vb, Vc** - Phase voltages

### Outputs (Fault Types):

- **G C B A** - Fault labels

### Fault Examples:

| G | C | B | A | Fault Type |
| -- | -- | -- | -- | ---------------------- |
| 0  | 0  | 0  | 0  | No Fault               |
| 1  | 0  | 0  | 1  | LG Fault (A-Gnd)       |
| 0  | 0  | 1  | 1  | LL Fault (A-B)         |
| 1  | 0  | 1  | 1  | LLG Fault (A-B-Gnd)    |
| 0  | 1  | 1  | 1  | LLL Fault (A-B-C)      |
| 1  | 1  | 1  | 1  | LLLG Fault (A-B-C-Gnd) |

## Model Training

To train the model, run:

```bash
python train_final.py
```

This script loads the dataset, trains the DGR-ELM model, and saves it as `dgr_elm_model.pkl`.

## Usage

To make fault predictions using the trained model:

```bash
python predict1.py
```

The script will take input values from new_fault_data.csv and store the predicted fault type in fault_predictions.csv.

## Results

The trained DGR-ELM model provides accurate fault classification. The results can be found in `fault_predictions.csv` and `new_fault_data.csv`, which store predicted values for various test inputs.

## Contributions

Feel free to contribute by submitting issues or pull requests.

## License

This project is open-source and available under the MIT License.

---

For more details, check out the code and dataset in this repository!

