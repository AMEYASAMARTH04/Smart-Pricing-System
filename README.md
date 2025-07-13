# Smart-Pricing-System

# importing libraries
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 
 # Machine Learning libraries
 from sklearn.model_selection import train_test_split
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.metrics import mean_squared_error
 import pandas as pd
 import numpy as np
 from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
 from sklearn.metrics import mean_squared_error, r2_score
 import matplotlib.pyplot as plt
 from xgboost import XGBRegressor

 # Reading File
 external_df= pd.read_csv(r"C:\Users\ameys\Downloads\walmart_dynamic_pricing_data\external_factors.csv")
 external_df

 sales_df= pd.read_csv(r"C:\Users\ameys\Downloads\walmart_dynamic_pricing_data\historical_sales.csv")
 sales_df

 inventory_df= pd.read_csv(r"C:\Users\ameys\Downloads\walmart_dynamic_pricing_data\inventory_data.csv")
 inventory_df

 # Checking For Null Values
 external_df.isnull().sum()
 sales_df.isnull().sum()
 inventory_df.isnull().sum()

 # Data Info
 external_df.info()
 sales_df.info()
 inventory_df.info()
 
 # filling Null 
 external_df = external_df.fillna('Regular Day')
 external_df

 # Date columns to datetime 
 sales_df['Date'] = pd.to_datetime(sales_df['Date'])
 inventory_df['ExpiryDate'] = pd.to_datetime(inventory_df['ExpiryDate'])
 external_df['Date'] = pd.to_datetime(external_df['Date'])
 
 # Check data types
 sales_df.dtypes

  # merging
 # Merge sales + inventory on ProductName and Category
 merged_df = pd.merge(
    sales_df,
    inventory_df,
    on=['ProductName', 'Category'],
    how='left'
 )
 # Merge with external factors on Date
 merged_df = pd.merge(
    merged_df,
    external_df,
    on='Date',
    how='left'
 )
 # Preview final merged data
 print(merged_df.head())

# Feature Engineering

# Sort for lag & rolling
 merged_df = merged_df.sort_values(by=['ProductName', 'Date'])
 # Lag & Rolling
 merged_df['SalesVolume_Lag1'] = merged_df.groupby('ProductName')['SalesVolume'].shift(1)
 merged_df['SalesVolume_Lag7'] = merged_df.groupby('ProductName')['SalesVolume'].shift(7)
 merged_df['SalesVolume_Rolling7'] = merged_df.groupby('ProductName')['SalesVolume'].shift(1).rolling(7).mean()
 # Days to Expiry
 merged_df['DaysToExpiry'] = (merged_df['ExpiryDate'] - merged_df['Date']).dt.days
 # Low stock flag
 merged_df['IsLowStock'] = (merged_df['CurrentStockLevel'] < 10).astype(int)
 # Price diff with competitor
 merged_df['PriceDiff_Competitor'] = merged_df['PricePerUnit'] - merged_df['CompetitorAveragePrice']
 # Event & Weather
 merged_df['HasEvent'] = merged_df['Event'].notnull().astype(int)
 merged_df['IsBadWeather'] = merged_df['Weather'].apply(lambda x: 1 if x in ['Rain', 'Snow'] else 0)
 # Date parts
 merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek
 merged_df['IsWeekend'] = (merged_df['DayOfWeek'] >= 5).astype(int)
 merged_df['Month'] = merged_df['Date'].dt.month
 # One-hot encode Category
 merged_df = pd.get_dummies(merged_df, columns=['Category'], drop_first=True)
 # Fill missing
 merged_df.fillna(0, inplace=True)
 print(merged_df.head())

 # Spliting x and y
 # Final features
 feature_cols = [
    'PricePerUnit',
    'SalesVolume_Lag1',
    'SalesVolume_Lag7',
    'SalesVolume_Rolling7',
    'CurrentStockLevel',
    'IsLowStock',
    'DaysToExpiry',
    'PriceDiff_Competitor',
    'HasEvent',
    'IsBadWeather',
    'DayOfWeek',
    'IsWeekend',
    'Month'
 ] + [col for col in merged_df.columns if col.startswith('Category_')]
 # X and y
 X = merged_df[feature_cols]
 y = merged_df['SalesVolume']
 # Split
 X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
 )
 print("Train shape:", X_train.shape)

 
# training Ml model
# XGBoost
 param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
 }

from sklearn.model_selection import GridSearchCV
 param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
 }
 grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
 )
 grid.fit(X_train, y_train)
 print("Best params:", grid.best_params_)
 print("Best CV RMSE:", -grid.best_score_)

  
# model Evalution
 from sklearn.model_selection import cross_val_score
 best_rf = grid.best_estimator_
 y_pred = best_rf.predict(X_test)
 rmse = mean_squared_error(y_test, y_pred, squared=False)
 r2 = r2_score(y_test, y_pred)
 print("Final Test RMSE:", rmse)
 print("Final Test R²:", r2)
 # Cross-validation check
 scores = cross_val_score(best_rf, X, y, cv=5, scoring='neg_root_mean_squared_error')
 print("5-Fold CV RMSE:", -scores.mean())


 # saving Predictions
 results_df = X_test.copy()
 results_df['Actual_SalesVolume'] = y_test.values
 results_df['Predicted_SalesVolume'] = y_pred
 results_df.to_csv('xgboost_dynamic_pricing_predictions.csv', index=False)
 print("Saved to 'xgboost_dynamic_pricing_predictions.csv'")


 
