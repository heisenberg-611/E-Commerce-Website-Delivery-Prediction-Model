# E-commerce Shipping Delivery Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview
This project aims to predict whether an e-commerce shipment will reach its destination on time using machine learning. Based on the **E-commerce Shipping Dataset**, the pipeline processes historical shipping data, explores patterns, and evaluates multiple supervised learning models to predict delivery delays. The target variable is `Reached.on.Time_Y.N`.

## Features and Workflow
1. **Data Pre-processing**: Handles categorical encoding (LabelEncoder) and feature scaling (StandardScaler) to prepare data for distance-based and gradient descent algorithms.
2. **Exploratory Data Analysis (EDA)**: Visualizes class distributions, feature relationships, and correlation matrices to identify key predictors like `Discount_offered`.
3. **Unsupervised Learning**: Applies K-Means clustering and PCA dimensionality reduction to visualize inherent structures within the dataset.
4. **Supervised Learning**: Trains and evaluates five distinct classification algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - Neural Network (Multi-Layer Perceptron)
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes
5. **Evaluation Strategy**: Compares models using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC curves.

## Repository Contents
- **`E-commerce Shipping Delivery Prediction.ipynb`**: The core Jupyter Notebook containing the full implementation, visualizations, and model training.
- **`documentation.md`**: A highly comprehensive, in-depth guide explaining the mathematical intuitions, business logic, algorithmic mechanics, and technical rationale behind every step of the project.
- **`E-commerce Shipping Dataset.csv`**: The dataset used for model training and evaluation.
- **`generate_notebook.py`**: A python script used to procedurally generate the Jupyter Notebook.

## Installation & Usage

1. **Clone the repository** (if applicable):
   ```bash
   git clone https://github.com/heisenberg-611/CSE422-Final-ML-Project-Spring26.git
   cd "CSE422 Final ML Project Spring26"
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nbformat
   ```

3. **Run the Notebook**:
   Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook "E-commerce Shipping Delivery Prediction.ipynb"
   ```
   *Alternatively, you can generate a fresh notebook by running `python generate_notebook.py`.*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
