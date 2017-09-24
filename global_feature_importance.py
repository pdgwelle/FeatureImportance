import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


## This is the change in y for a 1% change in x
## not obvious how this would apply for categorical variables
def global_marginal_effects(rf_model, X, y, epsilon_percent=0.01):
    # initialize output
    dydx = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    
    # adjustment is epsilon_percent of range since we will both add and subtract
    epsilon = epsilon_percent/2 * (X.max() - X.min())
    epsilon = pd.DataFrame(np.diag(epsilon))
    
    # populate dydx
    for i, e in epsilon.iterrows():
        X_hi = X + e
        X_lo = X - e
        y_hi = rf_model.predict(X_hi)
        y_lo = rf_model.predict(X_lo)
        dydx.loc[:, i] = y_hi - y_lo
        
    return dydx.mean()

## This is the marginal effect for varying from 0.05 to 0.95 percentile
## For categorical variable it is effect of varying from 0 to 1
def global_marginal_effects(rf_model, X, y):
    # initialize output
    dydx = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    
    # 5th and 95th percentile df
    fifth_percentile_X = X.quantile(q=0.05)
    ninety_fifth_percentile_X = X.quantile(q=0.95)

    # populate dydx
    for i, col in enumerate(X.columns):
        
        # Store old copy of X[col]
        X_old = X[col].copy()
        
        # Store max and min
        Xmax = X[col].max()
        Xmin = X[col].min()
        
        # If categorical set X_hi to 1
        if((Xmax == 1) & (Xmin == 0)): X[col] = 1
        else: X[col] = ninety_fifth_percentile_X[col]
        y_hi = rf_model.predict_proba(X)[:,1]
        
        # If categorical set X_lo to 0
        if((Xmax == 1) & (Xmin == 0)): X[col] = 0
        else: X[col] = fifth_percentile_X[col]
        y_lo = rf_model.predict_proba(X)[:,1]

        # Calculate dydx at each point for column
        dydx.loc[:, col] = y_hi - y_lo
        
        # Reset X[col]
        X[col] = X_old
        
    return dydx, dydx.mean()