# Regression-and-Evaluation
# California Housing Price Prediction - Regression and Evaluation

## Overview

This project focuses on predicting median house values in California using the California Housing dataset available through scikit-learn. It explores various regression techniques, evaluates their performance, and utilizes hyperparameter tuning to optimize the best-performing model.

## Dataset

The dataset used is the **California Housing dataset**, fetched using `sklearn.datasets.fetch_california_housing`.

* **Target Variable:** `MedHouseVal` (Median House Value)
* **Features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

## Workflow

1.  **Data Loading & Initial Exploration:**
    * Load the dataset into a pandas DataFrame.
    * Perform basic EDA: check head, description, info, shape, duplicates, and null values.
2.  **Data Preprocessing:**
    * **Outlier Handling:** Visualize outliers using boxplots and cap them using the Interquartile Range (IQR) method.
    * **Feature Scaling:** Scale the features using `StandardScaler` as all features are numerical.
    * **Correlation Analysis:** Visualize feature correlations using a heatmap.
    * **Data Splitting:** Split the data into training (67%) and testing (33%) sets.
3.  **Model Training & Base Evaluation:**
    * Train several regression models on the training data:
        * Linear Regression
        * Decision Tree Regressor
        * Random Forest Regressor
        * Gradient Boosting Regressor
        * Support Vector Regressor (SVR)
    * Evaluate each model on the test data using:
        * Mean Absolute Error (MAE)
        * Mean Squared Error (MSE)
        * R-squared (R²) Score
    * Identify the best-performing model based on these metrics (Random Forest Regressor showed the highest R² and lowest MAE/MSE initially).
4.  **Hyperparameter Tuning:**
    * Optimize the **Random Forest Regressor** using:
        * `GridSearchCV`: Exhaustively searches a predefined grid of hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) with 5-fold cross-validation to find the absolute best combination.
        * `RandomizedSearchCV`: Samples a fixed number of parameter settings from the specified distributions, offering a faster alternative to `GridSearchCV`.
    * Evaluate the tuned models obtained from both search methods using MAE, MSE, and R².
5.  **Results & Conclusion:**
    * Compare the performance of the models tuned with `GridSearchCV` and `RandomizedSearchCV`.

## Dependencies

* Python 3
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

You can install these using pip:
`pip install pandas numpy matplotlib seaborn scikit-learn`

## How to Run

1.  Ensure you have Python and the required libraries installed.
2.  Clone or download the repository containing the `Regression and Evaluation-Assignment.ipynb` file.
3.  Open the notebook using Jupyter Notebook or Jupyter Lab.
4.  Run the cells sequentially from top to bottom.

## Results Summary

* Initial model comparison indicated that **Random Forest Regressor** provided the best baseline performance (R² ≈ 0.805).
* Hyperparameter tuning was performed on the Random Forest Regressor using both `GridSearchCV` and `RandomizedSearchCV`.
* **GridSearchCV Result:** Achieved the slightly better performance with an R² score of approximately **0.807**.
    * MAE ≈ 0.3305
    * MSE ≈ 0.2478
* **RandomizedSearchCV Result:** Achieved a comparable performance with an R² score of approximately **0.806**, but finished execution significantly faster.
    * MAE ≈ 0.3316
    * MSE ≈ 0.2487

## Conclusion

The Random Forest Regressor, particularly when tuned using `GridSearchCV`, provides the best predictive performance for this dataset among the models tested. However, `RandomizedSearchCV` offers a good balance between performance and computational efficiency, yielding nearly identical results in much less time.
