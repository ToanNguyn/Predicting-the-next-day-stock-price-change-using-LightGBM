from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def split_time_series_data(df, target_col='Target', date_col='Date', test_size=0.2):
    df = df.copy()

    features = df.drop(columns=[target_col, date_col])
    target = df[target_col]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features, target, test_size=test_size, shuffle=False
    )

    return features, target, X_trainval, X_test, y_trainval, y_test



def time_series_cv(model, X, y, dates, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_scores = []
    val_preds = []
    val_trues = []
    val_dates = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        date_val = dates.iloc[y_val.index]['Date']

        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)

        val_scores.append(mean_absolute_error(y_val, y_val_pred))
        val_preds.extend(y_val_pred)
        val_trues.extend(y_val.values)
        val_dates.extend(date_val.values)

    return val_scores, val_preds, val_trues, val_dates