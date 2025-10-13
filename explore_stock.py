import pandas as pd
import matplotlib.pyplot as plt

print("Exploring The Stock Data ")

# Skip the complex headers and load from row 2
df = pd.read_csv("tesla_stock_data.csv", skiprows=2)

# Manually set column names based on what we saw
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert to proper types
df['Date'] = pd.to_datetime(df['Date'])
df['Close'] = pd.to_numeric(df['Close'])

# Set date as index
df.set_index('Date', inplace=True)

print("✅ Data loaded successfully!")
print("📊 Basic Info:")
print(df.info())
print("\n📈 First 5 Rows:")
print(df.head())

# Plot
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Tesla Closing Price', color='red', linewidth=2)
plt.title("Tesla Stock Price History")
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('tesla_price_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Chart saved as 'tesla_price_chart.png'")