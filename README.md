# Gradient_Descent_MLR_Implementation
Gradient Descent implementation for Multiple linear regression

---

# 📊 Multiple Linear Regression from Scratch (Gradient Descent)

This project demonstrates a basic implementation of **Multiple Linear Regression (MLR)** using **Gradient Descent**, without using `sklearn`'s regression models. It's designed for educational purposes to show how linear models work under the hood.

---

## 🚀 Features

* Implements multiple linear regression using gradient descent.
* Scales features using `StandardScaler` for better convergence.
* Custom training (`fit`) and prediction (`predict`) methods.
* Works on any dataset with a numeric target column named `income`.

---

## 🧠 Algorithm

The model optimizes weights using the **gradient descent algorithm**:

1. Initialize weights to zero.
2. Scale features using Z-score normalization.
3. Iteratively update weights using the gradient of the Mean Squared Error (MSE) loss function.

---

## 🛠️ Dependencies

* `numpy`
* `pandas`
* `scikit-learn` (for `StandardScaler`)

Install dependencies with:

```bash
pip install numpy pandas scikit-learn
```

---

## 📁 Dataset

The dataset should be a CSV file containing multiple numerical features and a target column named **`income`**.

Example path:

```
Multiple Linear Regression/multiple_linear_regression_dataset.csv
```

---

## 📌 Usage

```python
import pandas as pd
from mlr import MLR_Grad_Desc  # Assuming you've saved the class in a file named mlr.py

# Load data
data = pd.read_csv("Multiple Linear Regression/multiple_linear_regression_dataset.csv")

# Initialize and train the model
mlr = MLR_Grad_Desc(data, lr=0.01)
mlr.fit(epochs=300)

# View learned weights
print(mlr.wgt)

# Predict on new data
new_data = pd.DataFrame({...})  # Replace with actual feature data
predictions = mlr.predict(new_data)
```

---

## 📈 Output

The final weights learned by the model will be printed after training:

```bash
[weight_0, weight_1, weight_2, ..., weight_n]
```

---

## ⚠️ Notes

* Make sure the `income` column is present in your dataset.
* Feature scaling is crucial here, as gradient descent is sensitive to feature magnitudes.
* Always preprocess your new input data the same way as training data.

---

## ✅ TODO

* [ ] Add support for mini-batch gradient descent.
* [ ] Add MSE/loss tracking over epochs.
* [ ] Save and load trained weights.
* [ ] Visualize convergence.

---

## 📚 License

MIT License. Free to use and modify.

---

Let me know if you'd like me to generate the file structure or write a `setup.py` or test case for this project.
