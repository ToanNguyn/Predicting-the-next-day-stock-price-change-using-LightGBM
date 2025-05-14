import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt 
import seaborn as sns

def feature_engineering(df):
    # Time features
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek        
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])                  
    df['Month'] = df['Date'].dt.month    
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Cyclical encoding
    df['Day_sin'] = np.sin(2 * np.pi * df['Date'].dt.day/ 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Date'].dt.day/ 31)

    df['Month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month/ 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month/ 12)

    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    # Price-based features
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility_20'] = df['Return'].rolling(window=20).std()

    # Volume-based features
    df['Close_vs_MA5'] = df['Close'] - df['MA_5']
    df['MA5_vs_MA20'] = df['MA_5'] - df['MA_20']

    df['Volume_avg_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_spike'] = df['Volume'] / df['Volume_avg_5']

    # Intraday features
    df['Intraday_range'] = df['High'] - df['Low']
    df['Open_to_Close'] = df['Close'] - df['Open']

    # Technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(close=df['Close']).macd()
    df['BB_bbm'] = ta.volatility.BollingerBands(close=df['Close']).bollinger_mavg()

    df.dropna(inplace=True)

    return df

def aggregate_intraday_to_daily(df):
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y %H:%M')

    df = df.sort_values('Date/Time') 

    df.drop(['Open Interest', 'Ticker'], axis=1, inplace=True)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    df['Date'] = df['Date/Time'].dt.date
    df = df.groupby('Date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    df = df.reset_index()

    return df 

def remove_outliers_and_plot(df, column='Target'):
    plt.figure(figsize=(10, 2))
    sns.boxplot(x=df[column], color='orange')
    plt.title(f'Boxplot của biến động giá cổ phiếu ({column})')
    plt.grid(True)
    plt.show()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Số dòng còn lại sau khi loại outliers: {len(df_filtered)}")

    plt.figure(figsize=(10, 2))
    sns.boxplot(x=df_filtered[column], color='orange')
    plt.title(f'Boxplot của biến động giá cổ phiếu ({column}) sau khi loại outliers')
    plt.grid(True)
    plt.show()

    return df_filtered

def return_volatility_target(df):
    df['Return'] = df['Close'].pct_change()
    df['return_5'] = df['Close'].pct_change(5)

    df['volatility_5'] = df['Close'].pct_change().rolling(window=5).std()

    df['Target'] = df['Close'].shift(-1) - df['Close']

    return df