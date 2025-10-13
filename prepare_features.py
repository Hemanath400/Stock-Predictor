import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv("tesla_stock_data.csv", skiprows=2)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# 2. Fix data types
df['Date'] = pd.to_datetime(df['Date'])
numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# 3. Organize data
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
print(f"Original data {df.shape}")

# 4. Create target 
df['Tomorrow'] = df['Close'].shift(-1)
df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)  # FIXED LINE!

# 5. Create features 
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
df['MA_50'] = df['Close'].rolling(50).mean()

df['Volatility_5'] = df['Close'].rolling(5).std()
df['Volatility_20'] = df['Close'].rolling(20).std()

df['Price_Change_1d'] = df['Close'].pct_change(1)
df['Price_Change_5d'] = df['Close'].pct_change(5)
df['Volume_Change_1d'] = df['Volume'].pct_change(1)  # FIXED: Use Volume!

df['Vs_MA_5'] = (df['Close'] - df['MA_5']) / df['MA_5']
df['Vs_MA_20'] = (df['Close'] - df['MA_20']) / df['MA_20']  # FIXED: Use MA_20!

# 6. Clean up
df = df.dropna()
print(f"📊 After cleaning: {df.shape}")
print(f"🎯 Target distribution:\n{df['Target'].value_counts()}")

# 7. Prepare for ML - FIXED feature names to match what we created
features = ['Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 'Volatility_5', 'Volatility_20',
           'Price_Change_1d', 'Price_Change_5d', 'Volume_Change_1d', 'Vs_MA_5', 'Vs_MA_20']

X = df[features]
y = df['Target']

print(f"\n📈 Final dataset shape: {X.shape}")
print(f"🎯 Target distribution:\n{y.value_counts()}")

# 8. Save
df.to_csv('tesla_ml_ready.csv')
print(f"✅ Features prepared! Saved {len(features)} features to 'tesla_ml_ready.csv'")

# 9. Correlation analysis
correlation_with_target = df[features + ['Target']].corr()['Target'].sort_values(ascending=False)
print(f"\n📊 Feature correlations with target:")
print(correlation_with_target)