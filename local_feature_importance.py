import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

## This is the marginal effect for varying from 0.05 to 0.95 percentile
## For categorical variable it is effect of varying from 0 to 1
def local_marginal_effects(model, X, y):

    def determine_categorical_or_continuous(X):
        out_dict = {}
        for col in X.columns:
            if((X[col].max() == 1) & (X[col].min() == 0)):
                out_dict[col] = True
            else:
                out_dict[col] = False
        return out_dict

    def calculate_effects(row, model, mean_X, categorical_dict):
        row_indices = row.index
        row = row.values.reshape(1,-1)
        out_series = pd.Series(np.nan, index=row_indices)
        prediction = rf_model.predict_proba(row)[:,1]
        for i, col in enumerate(row_indices.values):
            row_old = row.copy()
            if(categorical_dict[col]):
                row[:,i] = 1-row[:,i]
                out_series[col] = prediction - model.predict_proba(row)[:,1]
            else:
                row[:,i] = mean_X[col]
                out_series[col] = prediction - model.predict_proba(row)[:,1]
            row = row_old
        return out_series

    dydx = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    mean_X = X.mean()
    categorical_dict = determine_categorical_or_continuous(X)
    out_df = X.apply(calculate_effects, axis=1, model=model, mean_X=mean_X, categorical_dict=categorical_dict)       

    return out_df
