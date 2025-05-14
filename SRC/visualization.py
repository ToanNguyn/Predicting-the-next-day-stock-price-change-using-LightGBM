import pandas as pd
import matplotlib.pylab as plt

def plot_test_predictions(dates, y_true, y_pred, title='Actual vs Predicted Values from LightGBM (Optuna)'):
    result_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_true,
        'Predicted': y_pred
    })

    plt.figure(figsize=(24, 5))
    plt.plot(result_df['Date'], result_df['Actual'], label='Actual', marker='o')
    plt.plot(result_df['Date'], result_df['Predicted'], label='Predicted (Optuna)', marker='x')
    plt.title(title)
    plt.xlabel('Time (Date)')
    plt.ylabel('Price Change (Target)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cv_predictions(dates, y_true, y_pred, title='Actual vs Predicted Values from LightGBM (Cross-validation)'):
    plt.figure(figsize=(24, 5))
    plt.plot(dates, y_true, label='Actual', marker='o')
    plt.plot(dates, y_pred, label='Predicted (CV)', marker='x')
    plt.title(title)
    plt.xlabel('Time (Date)')
    plt.ylabel('Price Change (Target)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()