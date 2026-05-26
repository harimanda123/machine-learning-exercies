# Machine Learning Exercises

A hands-on collection of machine learning and deep learning notebooks covering neural network architectures, unsupervised clustering, and explainable AI techniques — built with Keras, PyTorch, and scikit-learn.

---

## Repository Structure

```
machine-learning-exercies/
|-- Convolutional Neural Networks/
|   |-- FullyConnectedNeuralNetwork_Classification_Keras.ipynb
|   |-- FullyConnectedNeuralNetwork_Regression_PyTorch.ipynb
|   `-- RecurrelNeuralNetworks_GRU_Keras.ipynb
|-- Explainable AI/
|   |-- california_housing_train.csv
|   |-- model selection - xAI.ipynb
|   `-- Model Selection and xAI-2026.pdf
|-- K-Means Clustering/
|   |-- K_Means_Clustering.ipynb
|   |-- kmeans_clustering_activity_exercise.ipynb
|   |-- carprices_dataset.csv
|   `-- kmeans_census_data.csv
|-- README.md
`-- requirements.txt
```

---

## Folders and Files

### 1. Convolutional Neural Networks

This folder contains notebooks exploring neural network architectures for supervised learning tasks.

#### `FullyConnectedNeuralNetwork_Classification_Keras.ipynb`
Builds a fully connected (dense) neural network using **Keras** for multi-class classification tasks.
- Data preprocessing and encoding of categorical targets
- Model architecture with Dense and Dropout layers
- Compilation with categorical cross-entropy loss and Adam optimiser
- Training with validation split, accuracy and loss tracking
- Evaluation with confusion matrix and classification report

#### `FullyConnectedNeuralNetwork_Regression_PyTorch.ipynb`
Builds a fully connected neural network using **PyTorch** for continuous value regression.
- Dataset preparation and normalisation using PyTorch DataLoader
- Model definition using `nn.Module` with Linear layers and activation functions
- Custom training loop with MSE loss and SGD/Adam optimiser
- Loss curve visualisation across epochs
- Evaluation with RMSE and R-squared score

#### `RecurrelNeuralNetworks_GRU_Keras.ipynb`
Implements a **Gated Recurrent Unit (GRU)** network using **Keras** for sequential data modelling.
- Sequence data preparation and time-step windowing
- GRU layer stacking with return sequences
- Compilation with mean squared error loss for time series prediction
- Training and validation loss monitoring
- Prediction visualisation against actual values

---

### 2. Explainable AI

This folder focuses on model interpretability techniques — understanding *why* a model makes its predictions.

#### `model selection - xAI.ipynb`
Demonstrates explainability methods applied to machine learning models trained on the California Housing dataset.
- Model selection across multiple algorithms (Random Forest, Gradient Boosting, Linear Regression)
- Feature importance analysis using permutation importance and SHAP values
- SHAP summary plots, dependence plots, and force plots
- Comparison of model performance (RMSE, R-squared) alongside interpretability
- Key factors driving house price predictions

#### `california_housing_train.csv`
Training dataset used in the XAI notebook. Contains California census data with features such as median income, house age, average rooms, population, and geographic coordinates, with median house value as the target.

| Column | Description |
|--------|-------------|
| `longitude` / `latitude` | Geographic location |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` / `total_bedrooms` | Room counts |
| `population` | Block population |
| `households` | Number of households |
| `median_income` | Median income (in tens of thousands USD) |
| `median_house_value` | Target variable — median house value |
| `ocean_proximity` | Proximity to ocean (categorical) |

#### `Model Selection and xAI-2026.pdf`
Reference document covering the theoretical background of explainable AI, model selection strategies, and interpretation of SHAP-based explanations used in the notebook.

---

### 3. K-Means Clustering

This folder covers unsupervised learning through K-Means clustering applied to real-world datasets.

#### `K_Means_Clustering.ipynb`
Core notebook demonstrating K-Means clustering from theory to practice.
- Elbow method and silhouette score to determine optimal number of clusters (k)
- Standardisation of features using `StandardScaler`
- Cluster assignment and centroid visualisation
- 2D scatter plots coloured by cluster label
- Interpretation of cluster characteristics

#### `kmeans_clustering_activity_exercise.ipynb`
Activity-based exercise notebook for applying K-Means to a new dataset.
- Exploratory data analysis (EDA) on the exercise dataset
- Feature selection and preprocessing
- K-Means model fitting with scikit-learn
- Cluster evaluation and visualisation
- Guided exercises with tasks for hands-on practice

#### `carprices_dataset.csv`
Dataset used for K-Means clustering exercises containing car listing data.

| Column | Description |
|--------|-------------|
| `Make` / `Model` | Car manufacturer and model name |
| `Year` | Year of manufacture |
| `Engine HP` | Engine horsepower |
| `Engine Cylinders` | Number of cylinders |
| `highway MPG` / `city mpg` | Fuel efficiency |
| `MSRP` | Manufacturer suggested retail price |

#### `kmeans_census_data.csv`
Census-based dataset used for K-Means clustering, containing demographic and socioeconomic features suitable for population segmentation tasks.

---

## Frameworks and Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| TensorFlow / Keras | Deep learning model building | >= 2.13.0 |
| PyTorch | Neural network definition and training | >= 2.0.0 |
| scikit-learn | K-Means, preprocessing, metrics | >= 1.3.0 |
| pandas | Data loading and manipulation | >= 2.0.0 |
| numpy | Numerical computation | >= 1.24.0 |
| matplotlib | Plotting and visualisation | >= 3.7.0 |
| seaborn | Statistical visualisation | >= 0.12.0 |
| SHAP | Model explainability (XAI notebook) | latest |

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

Clone the repository:

```bash
git clone https://github.com/harimanda123/machine-learning-exercies.git
cd machine-learning-exercies
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install shap
```

Launch Jupyter:

```bash
jupyter notebook
```

---

## Topics Covered

- Supervised learning — regression and classification with neural networks
- Recurrent neural networks and sequential data modelling (GRU)
- Unsupervised learning — K-Means clustering and segmentation
- Explainable AI — SHAP values, feature importance, model interpretability
- Model selection and comparison across algorithms
- Data preprocessing, normalisation, and feature engineering
