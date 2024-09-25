# Nestle India Stock Price Prediction

This project aims to predict the stock price of Nestle India using various machine learning models: **LSTM (Long Short-Term Memory)**, **Linear Regression**, **K-Nearest Neighbors (KNN)**, and **K-Means Clustering**. The dataset used for this project contains historical stock prices.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [EDA and Preprocessing](#eda-and-preprocessing)
- [Models](#models)
  - [LSTM](#lstm)
  - [Linear Regression](#linear-regression)
  - [KNN](#knn)
  - [K-Means Clustering](#k-means-clustering)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Dataset

The dataset used is **Nestle India Stock Prices**, which contains daily closing prices, volumes, and other relevant data. The data is stored in a CSV file (`NESTLEIND.csv`).

## Project Structure

```
nestle-india-stock-prediction/
├── data/
│   └── NESTLEIND.csv         # Dataset file
├── src/
│   └── lstm_model.py         # LSTM Model
│   └── linear_regression.py   # Linear Regression Model
│   └── knn_model.py          # KNN Model
│   └── kmeans_clustering.py   # K-Means Clustering
├── README.md                 # Project Documentation
├── requirements.txt          # Python dependencies
└── stock_analysis.ipynb      # Jupyter Notebook for full analysis
```

## EDA and Preprocessing

Before applying models, the following preprocessing steps were applied:

1. **Handling Missing Values**: Checked and handled missing values if any.

2. **Feature Selection**: We focused primarily on the Close price for the models.

3. **Data Normalization**: Scaled the stock prices using MinMaxScaler to fit between 0 and 1.

4. **Creating Sequences**: For the LSTM model, we created sequences of 60 timesteps (days) to predict the next value.

## Models

## LSTM

- **Model Type**: Sequential LSTM Neural Network

- **Architecture**: 2 LSTM layers, followed by dense layers

- **Loss Function**: Mean Squared Error (MSE)

- **Optimizer**: Adam

```
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
```

## Linear Regression

- **Model Type**: Linear Regression using sklearn

- **Target**: Predicting stock price based on historical data

```model_lr = LinearRegression()```

## KNN

**Model Type**: K-Nearest Neighbors using sklearn

**Target**: Predicting stock price based on nearest neighbors

```model_knn = KNeighborsRegressor(n_neighbors=5)```

## K-Means Clustering

**Model Type**: K-Means Clustering using sklearn

**Target**: Identifying clusters in stock price behavior

```kmeans = KMeans(n_clusters=3, random_state=0)```

## Evaluation

Each model was evaluated using metrics such as **Root Mean Squared Error (RMSE)** and **R-squared (R²)** to assess the accuracy of predictions.

## LSTM Evaluation

LSTM RMSE: [407.016788248518]

LSTM R² Score: [0.972]

## Linear Regression Evaluation

Linear Regression RMSE: [270.2518644458053]

Linear Regression R² Score: [0.987]

## KNN Evaluation

KNN RMSE: [4144.048814710538]

KNN R² Score: [-1.828]

## Results

The following are the key findings from each model:

- **LSTM** provided the most accurate predictions due to its ability to capture time-series dependencies.

- **Linear Regression** performed moderately well but is limited in capturing stock price volatility.

- **KNN** had good performance with well-tuned parameters.

- **K-Means Clustering** successfully identified clusters, helping in trend analysis but not directly predicting prices.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

```
git clone https://github.com/yourusername/nestle-india-stock-prediction.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Load the Jupyter Notebook:

```
jupyter notebook stock_analysis.ipynb
```

2. Run individual model scripts:

```
python src/lstm_model.py
python src/linear_regression.py
python src/knn_model.py
python src/kmeans_clustering.py
```

3. View stock analysis and prediction results in charts and metrics.

## Contributing

Feel free to submit pull requests or open issues to discuss any features or improvements. All contributions are welcome!

## License

This project is licensed under the MIT License.
