

import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates






#pulling data
def pull_daily_market_data(ticker="GOOGL"):
    start_date = "2020-01-01"
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)['Close']
    qqq_data = yf.download("QQQ", start=start_date, end=end_date, auto_adjust=True)['Close']
    if isinstance(stock_data, pd.DataFrame): stock_data = stock_data.iloc[:, 0]
    if isinstance(qqq_data, pd.DataFrame): qqq_data = qqq_data.iloc[:, 0]
    cpi_data = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)

    df = pd.DataFrame({'Stock_Price': stock_data,'QQQ_Price': qqq_data})
    df = df.join(cpi_data).rename(columns={'CPIAUCSL': 'Inflation_CPI'})
    df['Inflation_CPI'] = df['Inflation_CPI'].ffill()
    df['PE_Ratio'] = df['Stock_Price'] / df['Stock_Price'].rolling(window=5).mean()
    #changing weight
    df['Stock_1w_Trend'] = df['Stock_Price'].pct_change(5)
    df['Stock_4w_Trend'] = df['Stock_Price'].pct_change(20)
    df['QQQ_1w_Trend'] = df['QQQ_Price'].pct_change(5)
    df['QQQ_4w_Trend'] = df['QQQ_Price'].pct_change(20)
    df['Target_5d_Fwd_Change'] = df['Stock_Price'].shift(-5) / df['Stock_Price'] - 1

    return df.dropna()








#seting input data
df = pull_daily_market_data("GOOGL")
features = ['Stock_1w_Trend', 'Stock_4w_Trend','QQQ_1w_Trend', 'QQQ_4w_Trend', 'Stock_Price', 'Inflation_CPI', 'PE_Ratio']
x = df[features]
y = df['Target_5d_Fwd_Change']
x_base_train = x.loc['2020-01-01':'2025-01-01']
y_base_train = y.loc['2020-01-01':'2025-01-01']
X_test = X.loc['2025-01-02':]
y_test = y.loc['2025-01-02':]









print("--- 2026 CONTINUOUS LEARNING & PREDICTION TEST ---")
prediction = []
actuals = []
rolling_prediction = [] # Stores the daily % change prediction to average them

x_dy_training = x_base_train.copy()
y_dy_training = y_base_train.copy()

for date, row in X_test.iterrows():
    model = RandomForestRegressor(n_estimators=1000, random_state=18, n_jobs=-1)
    model.fit(x_dy_training, y_dy_training)
    current_features = row.values.reshape(1, -1)
    predicted_change = model.predict(current_features)[0]

    rolling_prediction.append(predicted_change)
    if len(rolling_prediction) > 5:
        rolling_prediction.pop(0)

    avg_predicted_change = sum(rolling_prediction) / len(rolling_prediction)
    predicted_price = row['Stock_Price'] * (1 + avg_predicted_change)

    try:
        actual_future_price = y_test.loc[date] # y_test holds the shifted values
        actual_price_calc = row['Stock_Price'] * (1 + actual_future_price)
        actuals.append(actual_price_calc)
        prediction.append(predicted_price)
        print(f"Date: {date.strftime('%Y-%m-%d')} | Actual Price Today: ${row['Stock_Price']:.2f}")
        print(f"--> Avg Forecasted Change for next week: {avg_predicted_change:.2%}")
        print(f"--> PREDICTED price 1 week out: ${predicted_price:.2f}")
        print(f"--> ACTUAL price 1 week out:    ${actual_price_calc:.2f}\n")
    except KeyError:
        pass # End of dataset
    x_dy_training = pd.concat([x_dy_training, pd.DataFrame([row], columns=X.columns)])
    y_dy_training = pd.concat([y_dy_training, pd.Series([y.loc[date]], index=[date])])

if actuals:
    error = mean_absolute_error(actuals, prediction)
    print(f"Average Prediction Error for 2026: ${error:.2f} per share")

#prediction of known
latest_data = X.iloc[-1].values.reshape(1, -1)
future_change = model.predict(latest_data)[0]
final_predicted_price = X.iloc[-1]['Stock_Price'] * (1 + future_change)

print(f"*** BLIND PREDICTION FOR 1 WEEK FROM TODAY: ${final_predicted_price:.2f} ***")













#Graphing inputs prediction and result.
dates = X_test.index[:len(prediction)]

fig, ax1 = plt.subplots(figsize=(14, 8))

# prices
ax1.plot(dates, actuals, label='Actual Price (Future)', color='blue', linewidth=2)
ax1.plot(dates, prediction, label='Predicted Price (1-Week Out)', color='orange', linestyle='--', linewidth=2)
ax1.set_ylabel('Stock Price ($)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title('Algorithmic Prediction vs Reality (Unified Graph)', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)

# QQQ
ax2 = ax1.twinx()
ax2.plot(dates, X_test['QQQ_1w_Trend'][:len(dates)], label='QQQ 1-Week Trend', color='purple', alpha=0.4)
ax2.set_ylabel('QQQ Growth %', fontsize=12, color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Inflation
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(dates, X_test['Inflation_CPI'][:len(dates)], label='Inflation (CPI)', color='green', linestyle=':', alpha=0.6)
ax3.set_ylabel('Inflation CPI', fontsize=12, color='green')
ax3.tick_params(axis='y', labelcolor='green')

# Time
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig.autofmt_xdate()

# Combine all
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='upper left')

plt.tight_layout()
plt.show()
