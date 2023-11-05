import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras.backend as backend
import keras.layers as layers
import keras.models as models


RAW_PATH = '../data/volumes/'
CONCAT_PATH = '../data/concatenated/'

def plot_profile(year='', color='grey', labels=False, ylabel=False, title=False):
    total_volumes = []
    total_volumes_squared = []
    counter = []
    first_file = True
    for filename in os.listdir(RAW_PATH):
        if year and filename.split('_')[1][:4] != year: continue
        file_path = f'{RAW_PATH}{filename}'
        for n, line in enumerate(open(file_path)):
            line = line.strip()
            [_, volumes] = tuple(line.split(','))
            volumes = float(volumes)
            if first_file:
                total_volumes.append(volumes)
                total_volumes_squared.append(volumes**2)
                counter.append(1)
            else:
                total_volumes[n] += volumes
                total_volumes_squared[n] += volumes**2
                counter[n] += 1
        first_file = False
        total_volumes = np.array(total_volumes)
        total_volumes_squared = np.array(total_volumes_squared)
        counter = np.array(counter)
    profile = total_volumes/counter
    error = np.sqrt(total_volumes_squared/counter - (total_volumes/counter)**2)/np.sqrt(counter)
    t = np.arange(48)
    plt.errorbar(t, profile, yerr=error, fmt='o', capsize=5, color=color, markersize=4, label=year)
    if labels:
        plt.xlabel('minutes')
        plt.xticks([0, 2 * 4, 2 * 8, 2 * 12, 2 * 16, 2 * 20, 2 * 24], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'])
    else:
        plt.xlabel('')
        plt.xticks([], [])
    if ylabel:
        plt.ylabel('Total traded over the period')
    else:
        plt.ylabel('')
    if title:
        plt.title('Average daily profile in log scale')
    else:
        plt.title('')


P = 5

def load_data():
    data = pd.read_csv(f'{CONCAT_PATH}/BTCUSDT_20180101_20231101.csv', names=['timestamp', 'volumes'])
    data['datetime'] = pd.to_datetime(data.timestamp*1000000)
    data = data.set_index('datetime')
    data = data.drop('timestamp', axis=1)
    data = data.resample('30min').last()
    data = data.fillna(method='ffill')
    data = data.reset_index()
    data['minute'] = data.datetime.dt.minute + data.datetime.dt.hour * 60
    data = data[['datetime', 'minute', 'volumes']]
    data = data.set_index('datetime')
    data = data.sort_index()
    rolling_avg = data.groupby('minute')['volumes'].rolling(window='30D', min_periods=1).mean().reset_index()
    rolling_avg = rolling_avg.set_index('datetime')
    rolling_avg = rolling_avg.sort_index()
    rolling_avg = rolling_avg.rename(columns={ 'volumes': 'profile' })
    rolling_avg = rolling_avg.drop('minute', axis=1)
    data = data.join(rolling_avg)
    data['delta'] = data.volumes - data.profile
    data['delta_prev'] = data.delta.shift(1)
    dummies = pd.get_dummies(data.minute, prefix='bin', drop_first=True)
    data = pd.concat((data, dummies), axis=1)
    for p in range(1, P + 1):
        data[f'volumes_prev_{p}'] = data.volumes.shift(p)
        previous_dummies = dummies.shift(p).rename(columns={ column: f'{column}_prev_{p}' for column in dummies.columns })
        data = pd.concat((data, previous_dummies), axis=1)
    data = data.dropna(axis=0)
    return data


def r_squared(y_true, y_pred):
    SS_res =  backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - SS_res/SS_tot


def create_pseudo_linear_1(num_bins):
    # Inputs
    input_bins_current = layers.Input(shape=(num_bins,), name="Bins_Input_Current")  # Binary indicators for each bin at time t
    input_bins_prev = layers.Input(shape=(num_bins,), name="Bins_Input_Prev")  # Binary indicators for each bin at time t-1
    input_v_prev = layers.Input(shape=(1,), name="Previous_V")  # Previous value of V(t)
    # Shared layer for the same weights for I_b(t) and I_b(t-1)
    shared_dense = layers.Dense(1, activation='linear', use_bias=False, name="Shared_Weights")
    weighted_current = shared_dense(input_bins_current)
    weighted_prev = shared_dense(input_bins_prev)
    # COntribution of the previous delta to profile
    delta_layer = layers.Dense(1, activation='linear', use_bias=False, name="Delta_Layer")(input_v_prev - weighted_prev)
    # Sum up the components
    output_v = layers.Add(name="Output_V")([weighted_current, delta_layer])
    model = models.Model(inputs=[input_bins_current, input_bins_prev, input_v_prev], outputs=output_v)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_squared])
    return model


def create_pseudo_linear_P(num_bins, P=P):
    # Inputs
    input_bins_current = layers.Input(shape=(num_bins,), name="Bins_Input_Current")  # Binary indicators for each bin at time t
    input_bins_prev = { p: layers.Input(shape=(num_bins,), name=f'Bins_Input_Prev_{p}') for p in range(1, P + 1) }  # Binary indicators for each bin at time t-1
    input_v_prev = { p: layers.Input(shape=(1,), name=f'Previous_V_{p}') for p in range(1, P + 1) }  # Previous value of V(t)
    # Shared layer for the same weights for I_b(t) and I_b(t-1)
    shared_dense = layers.Dense(1, activation='linear', use_bias=False, name="Shared_Weights")
    weighted_current = shared_dense(input_bins_current)
    # Weighted previous value of V(t)
    delta_outputs = []
    # Loop to create layers and collect their outputs
    for p in range(1, P + 1):
        # The output of each layer is subtracted by another layer's output
        difference = layers.Subtract()([input_v_prev[p], shared_dense(input_bins_prev[p])])
        # A dense layer is applied to the difference
        delta = layers.Dense(1, activation='linear', use_bias=False, name=f'Delta_Layer_{p}')(difference)
        # Collect the output
        delta_outputs.append(delta)
    # If you want to sum the outputs of the created Dense layers
    delta_layer = layers.Add()(delta_outputs) if len(delta_outputs) > 1 else delta_outputs[0]
    # Sum up the components
    output_v = layers.Add(name="Output_V")([weighted_current, delta_layer])
    inputs = [ input_bins_current ]
    inputs += [ input_bins_prev[p] for p in range(1, P + 1) ]
    inputs += [ input_v_prev[p] for p in range(1, P + 1) ]
    model = models.Model(inputs=inputs, outputs=output_v)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_squared])
    return model


def create_LSTM(num_bins, P=P):
    # Inputs
    input_bins_current = layers.Input(shape=(num_bins,), name="Bins_Input_Current")  # Binary indicators for each bin at time t
    input_bins_prev = { p: layers.Input(shape=(num_bins,), name=f'Bins_Input_Prev_{p}') for p in range(1, P + 1) }  # Binary indicators for each bin at time t-1
    input_v_prev = { p: layers.Input(shape=(1,), name=f'Previous_V_{p}') for p in range(1, P + 1) }  # Previous value of V(t)
    # Shared layer for the same weights for I_b(t) and I_b(t-1)
    shared_dense = layers.Dense(1, activation='linear', use_bias=False, name="Shared_Weights")
    weighted_current = shared_dense(input_bins_current)
    # Weighted previous value of V(t)
    delta_outputs = []
    # Loop to create layers and collect their outputs
    for p in reversed(range(1, P + 1)):
        # The output of each layer is subtracted by another layer's output
        delta = layers.Subtract()([input_v_prev[p], shared_dense(input_bins_prev[p])])
        # Collect the output
        delta_outputs.append(delta)
    delta_outputs = backend.stack(delta_outputs, axis=1)
    # If you want to sum the outputs of the created Dense layers
    # x = layers.LSTM(15, return_sequences=False)(delta_outputs)
    x = layers.LSTM(100, activation='relu', return_sequences=True)(delta_outputs)
    x = layers.Dropout(.1)(x)
    x = layers.LSTM(50, activation='relu', return_sequences=False)(x)
    x = layers.Dropout(.1)(x)
    x = layers.Dense(1, activation='linear')(x)
    # Sum up the components
    output_v = layers.Add(name="Output_V")([weighted_current, x])
    inputs = [ input_bins_current ]
    inputs += [ input_bins_prev[p] for p in range(1, P + 1) ]
    inputs += [ input_v_prev[p] for p in range(1, P + 1) ]
    model = models.Model(inputs=inputs, outputs=output_v)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_squared])
    return model