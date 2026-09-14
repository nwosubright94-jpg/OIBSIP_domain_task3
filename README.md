# Car Price Prediction with Machine Learning

## Objective
The goal of this project is to build and evaluate machine learning regression models to predict used car selling prices based on key vehicle attributes such as age, original present price, driven kilometers, fuel type, transmission type, and seller type.

## Tools Used
* Python
* Jupyter Notebook
* Pandas (Data manipulation and cleaning)
* NumPy (Numerical operations)
* Seaborn & Matplotlib (Exploratory data analysis and visualization)
* Scikit-Learn (Feature processing, regression algorithms, and performance metrics)

## Steps Performed
1. Data Loading & Inspection: Loaded the CarDekho dataset (`car data.csv`) using Pandas and checked structural properties, summary statistics, and null values.
2. Feature Engineering: Created a new feature `Age` by subtracting the manufacturing `Year` from the current year, then dropped redundant columns.
3. Exploratory Data Analysis (EDA): Visualized distributions and relationships between features like `Present_Price`, `Kms_Driven`, and target variable `Selling_Price`.
4. Preprocessing & Encoding: Converted categorical variables (`Fuel_Type`, `Seller_Type`, `Transmission`) into numerical representations using One-Hot Encoding (`pd.get_dummies`).
5. Train-Test Split: Divided the dataset into training (80%) and testing (20%) sets.
6. Model Building & Training: Trained Linear Regression and Random Forest Regressor models on the training data.
7. Model Evaluation: Computed key regression metrics on the test dataset, including Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared score ($R^2$).

## Key Outcomes & Results
* Feature Influence: `Present_Price` and vehicle `Age` proved to be the strongest predictors of used car selling prices.
* Random Forest Regressor: Delivered high accuracy with an $R^2$ score above 90%, capturing non-linear relationships and interactions effectively.
* Linear Regression: Provided a strong baseline model, accurately capturing overall pricing trends across standard features.
