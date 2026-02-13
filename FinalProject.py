import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda, Attention
import random
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import CSVLogger

# %% Import Data
df = pd.read_csv('8-Site_CA-PV1-DB-CA-1A.csv')

# %% Define Feature Cols
feature_cols = ['Wind_Speed',
                'Weather_Temperature_Celsius',
                'Global_Horizontal_Radiation',
                'Wind_Direction',
                'Weather_Daily_Rainfall',
                'Max_Wind_Speed',
                'Air_Pressure',
                'Pyranometer_1']

target_cols = ['Active_Power','Temperature_Probe_Avg']

df['Temperature_Probe_Avg'] = (df['Temperature_Probe_1'] + df['Temperature_Probe_2']) / 2
df = df.drop(['Temperature_Probe_1', 'Temperature_Probe_2'], axis=1)
df = df[['timestamp'] + feature_cols + target_cols]

# %%
# fix known GHR sensor accuracy issue
df.loc[df['Global_Horizontal_Radiation'] > 1400, 'Global_Horizontal_Radiation'] = 300
df.loc[df['Pyranometer_1'] > 2000, 'Pyranometer_1'] = 500
    
# %% Fill in missing data
df["datetime"] = pd.to_datetime(df["timestamp"])
df = df.sort_values('datetime').reset_index(drop=True)

full_index = pd.date_range( 
    start=df['datetime'].min(),
    end=df['datetime'].max(),
    freq='5min'
)

df = df.set_index('datetime').reindex(full_index).reset_index()
df = df.rename(columns={'index': 'datetime'})

df[feature_cols + target_cols] = df[feature_cols + target_cols].interpolate(
    method='linear',
    limit=12,                     
    limit_direction='both'
)

df = df.dropna(subset=feature_cols + target_cols)

# %%
df["date"]       = df["datetime"].dt.strftime("%Y-%m-%d")
df["time"]       = df["datetime"].dt.strftime("%H:%M:%S")
df["year"]       = df["datetime"].dt.year
df["month"]      = df["datetime"].dt.month
df["day"]        = df["datetime"].dt.day
df["hour_of_day"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
df['day_of_year'] = df['datetime'].dt.dayofyear

years = sorted(df['year'].unique())

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# # Remove Nighttime hours
df = df[(df['hour_of_day'] >= 4) & (df['hour_of_day'] <= 21)]

# Remove April 2016 As data is not complete
df = df[~((df['year'] == 2016))]

# Remove 2020 as it has bad data
df = df[~((df['year'] == 2020))]

# Remove All of 2025 and 2026 data due to data quality issues. features_AP
df = df[~((df['year'] >= 2026))]
# %%
# First loop: July 2019ily plots (all in one figure)
fig, axes = plt.subplots(5, 2, figsize=(20, 20))
axes = axes.flatten()

for idx, col in enumerate(feature_cols + target_cols):
    for day in range(1,32):
        axes[idx].plot(df[(df['year'] == 2019) & (df['month'] == 7) & (df['day'] == day)]['hour_of_day'],
                       df[(df['year'] == 2019) & (df['month'] == 7) & (df['day'] == day)][col],
                       color=(random.random(), random.random(), random.random()),
                       linewidth=0.8,
                       linestyle='-',       
                       marker=None            
                )
    axes[idx].set_title(f"{col} - July 2019 (one line per day)")
    axes[idx].set_xlabel("hour")
    axes[idx].set_ylabel(col)
    axes[idx].set_xlim(0, 24)
    axes[idx].set_xticks(range(0, 25, 2))
plt.tight_layout()
plt.savefig('fig_daily_plots.png', dpi=300, bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(5, 2, figsize=(20, 20))
axes = axes.flatten()

for idx, col in enumerate(feature_cols + target_cols):
    for year in years:
        year_data = df[df['year'] == year]
        avg_by_month = year_data.groupby('month')[col].mean().reset_index()
        
        axes[idx].plot(avg_by_month['month'],
                       avg_by_month[col],
                       color=(random.random(), random.random(), random.random()),
                       linewidth=1.5,
                       linestyle='-',       
                       marker='o',
                       label=str(year)
                ) 
    axes[idx].set_title(f"{col} - Monthly Averages (one line per year)")
    axes[idx].set_xlabel("Month")
    axes[idx].set_ylabel(col)
    axes[idx].set_xlim(1, 12)
    axes[idx].set_xticks(range(1, 13))
    axes[idx].set_xticklabels(month_names)
    axes[idx].legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
plt.tight_layout()
plt.savefig('fig_monthly_plots.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
# Cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df['year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
df['wind_dir_sin'] = np.sin(2 * np.pi * df['Wind_Direction'] / 360)
df['wind_dir_cos'] = np.cos(2 * np.pi * df['Wind_Direction'] / 360)

roll_windows = [6, 12, 24]

for window in roll_windows:
    df[f'GHI_roll_mean_{window}'] = df['Global_Horizontal_Radiation'].rolling(window=window, min_periods=1, center=False).mean()
    df[f'GHI_roll_std_{window}'] = df['Global_Horizontal_Radiation'].rolling(window=window, min_periods=1, center=False).std()
    df[f'PYR_roll_mean_{window}'] = df['Pyranometer_1'].rolling(window=window, min_periods=1, center=False).mean()
    df[f'PYR_roll_std_{window}'] = df['Pyranometer_1'].rolling(window=window, min_periods=1, center=False).std()
    df[f'WTC_roll_mean_{window}'] = df['Weather_Temperature_Celsius'].rolling(window=window, min_periods=1, center=False).mean()
    df[f'WTC_roll_std_{window}'] = df['Weather_Temperature_Celsius'].rolling(window=window, min_periods=1, center=False).std()

diff_windows = [5,15,30]
for d in diff_windows:
    df[f'GHI_diff_{d}'] = df['Global_Horizontal_Radiation'].diff(d)
    df[f'PYR_diff_{d}'] = df['Pyranometer_1'].diff(d)
    df[f'WTC_diff_{d}'] = df['Weather_Temperature_Celsius'].diff(d)

df = df.dropna()

# %%
df = df.set_index('datetime')
df = df.drop(["timestamp", "date", "time", "year", "month", "day",
              "hour_of_day", "day_of_year"], axis=1)

features_all = [c for c in df.columns if c not in ["Active_Power", "Temperature_Probe_Avg","Wind_Direction"]]

# %% Create LSTM to Predict Active Power using 2 hours of History
X_raw = df[features_all].values
y_raw = df[['Active_Power']].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

lookback = 24
X_seq, y_seq = [], []

for i in range(lookback, len(X_scaled)):
    if (df.index[i] - df.index[i - lookback] == pd.Timedelta(hours=2)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y_scaled[i])

X = np.array(X_seq)
y = np.array(y_seq)

train_len = int(len(X) * 0.7)
val_len = int(len(X) * 0.25)
X_train, X_val, X_test = X[:train_len], X[train_len:train_len + val_len], X[train_len + val_len:]
y_train, y_val, y_test = y[:train_len], y[train_len:train_len + val_len], y[train_len + val_len:]


inputs = Input(shape=(lookback, len(features_all)))
x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.3)(x)
attention_layer = Attention(use_scale=True)
x = attention_layer([x, x])
x = LSTM(32)(x)
x = Dropout(0.3)(x)
x = LSTM(32)(x)
x = Dropout(0.3)(x)
# x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = Dense(8, activation='relu')(x)
outputs = Dense(1)(x)


model1 = Model(inputs=inputs, outputs=outputs)
model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Increased learning rate
    loss='mse', 
    metrics=['mae']
)
model1.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

csv_logger = CSVLogger('training_log_model1.csv')

history1 = model1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    shuffle=False,
    callbacks=[early_stop,csv_logger],
    verbose=1
)

plt.figure(figsize=(10, 6))
plt.plot(history1.history['loss'], label='Train')
plt.plot(history1.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('fig_model_loss_power.png', dpi=300, bbox_inches='tight')
plt.show()


# Evaluation on test set
y_pred_scaled = model1.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_true = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
r2 = r2_score(y_test_true, y_pred)

model1.save('model_power_forecast.keras')

metrics1_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [mae, rmse, r2]
})

# Save to CSV
metrics1_df.to_csv('model1_metrics.csv', index=False)
print(metrics1_df)

# Sample prediction plot (first day in test set)
sample_point = random.randint(288, 10000)
plt.figure(figsize=(14, 6))
plt.plot(y_test_true[(sample_point-288):sample_point], label='Actual Active Power', linewidth=1.5)
plt.plot(y_pred[(sample_point-288):sample_point], label='Predicted Active Power', linewidth=1.5)
plt.title('Active Power vs Actual')
plt.xlabel('Time steps (5-minute intervals)')
plt.ylabel('Active Power')
plt.legend()
plt.grid(True)
plt.savefig('fig_model_fit_power.png', dpi=300, bbox_inches='tight')
plt.show()


# %% Create LSTM to Predict Temperature Probe using 2 hours of History
X_raw = df[features_all].values
y_raw = df[['Temperature_Probe_Avg']].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

lookback = 24
lookforward = 6 
X_seq, y_seq = [], []

for i in range(lookback, len(X_scaled)-lookforward):
    if (df.index[i] - df.index[i - lookback] == pd.Timedelta(hours=2)):
        if (df.index[i + lookforward] - df.index[i] == pd.Timedelta(minutes=30)):
            X_seq.append(X_scaled[i - lookback:i])
            y_seq.append([y_scaled[i][0], y_scaled[i + lookforward][0]])

X = np.array(X_seq)
y = np.array(y_seq)

train_len = int(len(X) * 0.7)
val_len = int(len(X) * 0.25)
X_train, X_val, X_test = X[:train_len], X[train_len:train_len + val_len], X[train_len + val_len:]
y_train, y_val, y_test = y[:train_len], y[train_len:train_len + val_len], y[train_len + val_len:]


inputs = Input(shape=(lookback, len(features_all)))
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
x = Dropout(0.3)(x)
x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.3)(x)
attention_layer = Attention(use_scale=True)
x = attention_layer([x, x])
x = LSTM(64)(x)
x = Dropout(0.3)(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
outputs = Dense(2)(x)


model2 = Model(inputs=inputs, outputs=outputs)
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Increased learning rate
    loss='mse', 
    metrics=['mae']
)
model2.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

csv_logger = CSVLogger('training_log_model2.csv')

history2 = model2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    shuffle=False,
    callbacks=[early_stop,csv_logger],
    verbose=1
)

plt.figure(figsize=(10, 6))
plt.plot(history2.history['loss'], label='Train')
plt.plot(history2.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('fig_model_loss_temp.png', dpi=300, bbox_inches='tight')
plt.show()


# Evaluation on test set
y_pred_scaled = model2.predict(X_test)

y_pred_current = scaler_y.inverse_transform(y_pred_scaled[:, 0:1])
y_pred_30min = scaler_y.inverse_transform(y_pred_scaled[:, 1:2])

y_test_current = scaler_y.inverse_transform(y_test[:, 0:1])
y_test_30min = scaler_y.inverse_transform(y_test[:, 1:2])

mae_current = mean_absolute_error(y_test_current, y_pred_current)
rmse_current = np.sqrt(mean_squared_error(y_test_current, y_pred_current))
r2_current = r2_score(y_test_current, y_pred_current)

mae_30min = mean_absolute_error(y_test_30min, y_pred_30min)
rmse_30min = np.sqrt(mean_squared_error(y_test_30min, y_pred_30min))
r2_30min = r2_score(y_test_30min, y_pred_30min)

model2.save('model_temp_forecast.keras')

metrics2_df = pd.DataFrame({
    'Timeframe':  ['Current','Current','Current','30 Minutes In the Future','30 Minutes In the Future','30 Minutes In the Future'],
    'Metric': ['MAE', 'RMSE', 'R²','MAE', 'RMSE', 'R²'],
    'Value': [mae_current, rmse_current, r2_current, mae_30min, rmse_30min, r2_30min]
})

# Save to CSV
metrics2_df.to_csv('model2_metrics.csv', index=False)
print(metrics2_df)

sample_point = random.randint(288, 10000)

plt.figure(figsize=(14, 6))
plt.plot(y_test_current[(sample_point-288):sample_point], label='Panel Temperature ', linewidth=1.5)
plt.plot(y_pred_current[(sample_point-288):sample_point], label='Predicted Panel Temperature ', linewidth=1.5)
plt.title('Panel Temperature Predicted vs Actual')
plt.xlabel('Time steps (5-minute intervals)')
plt.ylabel('Panel Temperature')
plt.legend()
plt.grid(True)
plt.savefig('fig_model_fit_temp_1.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(y_test_30min[(sample_point-288):sample_point], label='Panel Temperature 30 Minutes in the Future', linewidth=1.5)
plt.plot(y_pred_30min[(sample_point-288):sample_point], label='Predicted Panel Temperature 30 Minutes in the Future', linewidth=1.5)
plt.title('Panel Temperature Predicted vs Actual')
plt.xlabel('Time steps (5-minute intervals)')
plt.ylabel('Panel Temperature')
plt.legend()
plt.grid(True)
plt.savefig('fig_model_fit_temp_30_1.png', dpi=300, bbox_inches='tight')
plt.show()



# %%


# %%


# %%


# %%

