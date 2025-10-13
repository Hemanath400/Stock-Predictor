# 4_train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🤖 Training Stock Prediction Model...")

# 1. Load the prepared data
df = pd.read_csv('tesla_ml_ready.csv', parse_dates=['Date'], index_col='Date')

# 2. Define our features (clues) and target (answer)
features = ['Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 'Volatility_5', 'Volatility_20',
           'Price_Change_1d', 'Price_Change_5d', 'Volume_Change_1d', 'Vs_MA_5', 'Vs_MA_20']

x = df[features]  # Clues
y = df['Target']  # Answers (1=UP, 0=DOWN)

print(f"📊 Dataset: {x.shape[0]} days, {x.shape[1]} features")

# 3. Split data: 80% for training, 20% for testing
split_point = int(len(x) * 0.8)
x_train, x_test = x[:split_point], x[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"🛈 Training set: {x_train.shape} (First 80% of data)")
print(f"🧪 Test set: {x_test.shape} (Last 20% of data)")

# 4. Create and train the AI model
print("\n🎯 Training Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=100,  # Number of decision trees
    random_state=42,   # For reproducible results
    max_depth=10       # Prevent overfitting
)

# Train the model (this is where the AI learns!)
model.fit(x_train, y_train)

# 5. Make predictions
train_predictions = model.predict(x_train)  # Predict on training data
test_predictions = model.predict(x_test)    # Predict on test data

# 6. Evaluate performance
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"🏋️ Training Accuracy: {train_accuracy:.2%}")
print(f"🧪 Test Accuracy: {test_accuracy:.2%}")

# 7. Detailed performance report
print(f"\n📈 DETAILED REPORT:")
print(classification_report(y_test, test_predictions, target_names=['DOWN', 'UP']))

# 8. Feature Importance
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 TOP 5 MOST IMPORTANT FEATURES:")
for i, row in importance.head(5).iterrows():
    print(f"#{i+1}: {row['feature']} ({row['importance']:.2%})")

# 9. Save the model
# 9. Save the model
joblib.dump(model, 'stock_predictor_model.pkl')  # FIXED: .pkl not .pk1
print("💾 Model saved as 'stock_predictor_model.pkl'")  # FIXED: Removed extra line

show_prediction=True

if show_prediction:
    print(f"\n recent prdeiction(Sample of 5)")
    recent_data=x_test.tail(5)
    recent_predictions=model.predict(recent_data)

    for i,(idx,row)in enumerate(recent_data.iterrows()):
        prediction="UP" if recent_predictions[i]==1 else "DOWN"
        actual="UP" if y_test.loc[idx]==1 else "DOWN"
        correct="✅" if recent_predictions[i]==y_test.loc[idx] else  "❌"
        print(f"{idx.date()}: Predicted {prediction} | Actual: {actual} {correct}")