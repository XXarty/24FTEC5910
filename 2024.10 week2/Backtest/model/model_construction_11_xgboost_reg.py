import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def train_and_test(train_X, train_Y, test_X, test_Y):
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    parameters = {'n_estimators': [100, 200],
                  'max_depth': [3, 5],
                  'learning_rate': [0.01, 0.1],
                  'subsample': [1.0],
                  'colsample_bytree': [1.0]}

    grid = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
    grid.fit(train_X, train_Y.values.ravel())

    best_params = grid.best_params_
    model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    model.fit(train_X, train_Y.values.ravel())

    # Predictions
    Y_train_pred = model.predict(train_X)
    Y_test_pred = model.predict(test_X)

    # Calculate metrics
    mse = mean_squared_error(test_Y, Y_test_pred)
    mae = mean_absolute_error(test_Y, Y_test_pred)
    r2 = r2_score(test_Y, Y_test_pred)

    # Return the results
    return mse, mae, r2, Y_train_pred, Y_test_pred


def model_construction_xgboost(train_df, test_df, Y, X, Y_train, X_train, Y_test, X_test):
    results = []

    for y_col in Y:
        train_Y = Y_train[[y_col]]
        test_Y = Y_test[[y_col]]
        
        for x_col in X:
            train_X = X_train[[x_col]]
            test_X = X_test[[x_col]]
            mse, mae, r2, Y_train_pred, Y_test_pred = train_and_test(train_X, train_Y, test_X, test_Y)
            test_df[f'{x_col}___{y_col}_pred_rtn'] = Y_test_pred
            
            results.append((y_col, x_col, mse, mae, r2))
            
    results_df = pd.DataFrame(results, columns=['Y', 'X', 'MSE', 'MAE', 'R2'])
    result_df = results_df.sort_values(by='MSE').head(5)
    
    return result_df, test_df