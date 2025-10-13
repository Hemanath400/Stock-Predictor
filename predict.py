# 5_predict.py
import pandas as pd
import joblib

print("🔮 Making Tesla Stock Predictions...")
print("=" * 50)

# Load the trained model
model = joblib.load('stock_predictor_model.pkl')
print("✅ Model loaded successfully!")

# Load the latest data
df = pd.read_csv('tesla_ml_ready.csv', parse_dates=['Date'], index_col='Date')

features = ['Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 'Volatility_5', 'Volatility_20',
           'Price_Change_1d', 'Price_Change_5d', 'Volume_Change_1d', 'Vs_MA_5', 'Vs_MA_20']



# Get the most recent data 
latest = df.iloc[-1]  # Most recent trading day

from datetime import datetime

today = datetime.now().date()
print(f"🕒 Today's Date: {today}")
print(f"📊 Data Date: {latest.name.date()}")


print(f"✅ Prediction for: {latest.name.date()} → Next Trading Day")

print(f"\n📈 LATEST MARKET DATA (as of {latest.name.date()}):")
print(f"💰 Price: ${latest['Close']:.2f}")
print(f"📊 Volume: {latest['Volume']:,.0f} shares")
print(f"📈 5-Day Average: ${latest['MA_5']:.2f}")
print(f"📈 20-Day Average: ${latest['MA_20']:.2f}")
print(f"🎯 Price vs 5-Day Avg: {latest['Vs_MA_5']:+.2%}")

# Prepare data for prediction 
latest_features = latest[features].values.reshape(1, -1)

# Make prediction
prediction = model.predict(latest_features)[0]
probability = model.predict_proba(latest_features)[0]

prob_up = probability[1] * 100  # Probability of UP
prob_down = probability[0] * 100  # Probability of DOWN

print(f"\n🎯 TOMORROW'S PREDICTION:")
if prediction == 1:
    print(f"🟢 PRICE WILL GO UP ({prob_up:.1f}% confidence)")
else:
    print(f"🔴 PRICE WILL GO DOWN ({prob_down:.1f}% confidence)")

print(f"\n📊 CONFIDENCE LEVELS:")
print(f"🟢 UP: {prob_up:.1f}%")
print(f"🔴 DOWN: {prob_down:.1f}%")

# Trading suggestion
if prob_up > 70:
    print("\n💡 TRADING SUGGESTION: STRONG BUY SIGNAL! 📈")
elif prob_up > 60:
    print("\n💡 TRADING SUGGESTION: BUY SIGNAL 📈")
elif prob_down > 70:
    print("\n💡 TRADING SUGGESTION: STRONG SELL SIGNAL! 📉")
elif prob_down > 60:
    print("\n💡 TRADING SUGGESTION: SELL SIGNAL 📉")
else:
    print("\n💡 TRADING SUGGESTION: HOLD (Market uncertain) ⏸️")

print(f"\n🤖 AI MODEL STATS:")
print(f"✅ Accuracy: 59.14% (Beats random guessing!)")
print(f"✅ Trained on: {len(df)} days of historical data")
print(f"✅ Most important factor: Trading Volume")
print(f"✅ Prediction for: {latest.name.date()} → Tomorrow")

print("\n" + "=" * 50)
print("🎉 PREDICTION COMPLETE!")

print("=" * 50)
