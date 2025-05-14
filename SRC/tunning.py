from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import lightgbm as lgb

def optuna_objective(X, y, loss_type="rmse_corr", n_splits=3):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbosity': -1
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        losses = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**param)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            if loss_type == "rmse_corr":
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                corr = np.corrcoef(y_val, y_pred)[0, 1]
                loss = rmse - 0.5* corr
            elif loss_type == "mae":
                loss = mean_absolute_error(y_val, y_pred)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            losses.append(loss)

        return np.mean(losses)

    return objective
