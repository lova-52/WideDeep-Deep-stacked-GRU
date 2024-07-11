import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction import FeatureHasher
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import GRU, Dense, Dropout, LayerNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import time

warnings.filterwarnings('ignore')

# Load datasets
kddcup_99_path = 'kddcup.csv'
unsw_nb15_paths = ['UNSW-NB15_1.csv']
#unsw_nb15_paths = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']

kddcup_99 = pd.read_csv(kddcup_99_path)
unsw_nb15 = pd.concat([pd.read_csv(path) for path in unsw_nb15_paths])

# Preprocessing function for KDDCup99 dataset
def preprocess_kddcup(kddcup_99):
    numerical_features = [
        'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    categorical_features = ['protocol_type', 'service', 'flag']
    label_feature = 'label'

    # Encode label feature
    label_encoder = LabelEncoder()
    kddcup_99[label_feature] = label_encoder.fit_transform(kddcup_99[label_feature])

    # Process categorical features with FeatureHasher
    hasher = FeatureHasher(input_type='string', n_features=15)
    for feature in categorical_features:
        hashed_features = hasher.transform(kddcup_99[feature].astype(str).values.reshape(-1, 1)).toarray()
        hashed_feature_names = [f"{feature}_{i}" for i in range(hashed_features.shape[1])]
        hashed_df = pd.DataFrame(hashed_features, columns=hashed_feature_names)
        kddcup_99 = pd.concat([kddcup_99, hashed_df], axis=1).drop(columns=[feature])

    # Normalize numerical features (MinMaxScaler)
    scaler = MinMaxScaler()
    kddcup_99[numerical_features] = scaler.fit_transform(kddcup_99[numerical_features])

    # Apply PCA to reduce numerical features to 15 components
    pca = PCA(n_components=15)
    kddcup_99_pca = pca.fit_transform(kddcup_99[numerical_features])
    kddcup_99_pca_df = pd.DataFrame(kddcup_99_pca, columns=[f'pca_{i}' for i in range(15)])

    # Combine PCA features with other features
    final_df = pd.concat([kddcup_99.drop(columns=numerical_features), kddcup_99_pca_df], axis=1)

    return final_df

# Preprocessing function for UNSW-NB15 dataset
def preprocess_unsw_nb15(unsw_nb15):
    numerical_features = [
        'sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 
        'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
    ]
    categorical_features = ['proto', 'service', 'state']
    label_feature = 'label'

    # Convert 'sport' and 'dsport' to numeric and drop rows with non-numeric values
    unsw_nb15['sport'] = pd.to_numeric(unsw_nb15['sport'], errors='coerce')
    unsw_nb15['dsport'] = pd.to_numeric(unsw_nb15['dsport'], errors='coerce')
    unsw_nb15.dropna(subset=['sport', 'dsport'], inplace=True)

    # Encode attack category feature
    label_encoder = LabelEncoder()
    unsw_nb15['attack_cat'] = label_encoder.fit_transform(unsw_nb15['attack_cat'])

    # Encode label feature
    label_encoder = LabelEncoder()
    unsw_nb15[label_feature] = label_encoder.fit_transform(unsw_nb15[label_feature])

    # Process categorical features with FeatureHasher
    hasher = FeatureHasher(input_type='string', n_features=15)
    for feature in categorical_features:
        hashed_features = hasher.transform(unsw_nb15[feature].astype(str).values.reshape(-1, 1)).toarray()
        hashed_feature_names = [f"{feature}_{i}" for i in range(hashed_features.shape[1])]
        hashed_df = pd.DataFrame(hashed_features, columns=hashed_feature_names)
        unsw_nb15 = pd.concat([unsw_nb15, hashed_df], axis=1).drop(columns=[feature])

    # Drop rows with any NaN values in the numerical features
    unsw_nb15.dropna(subset=numerical_features, inplace=True)

    # Normalize numerical features (MinMaxScaler)
    scaler = MinMaxScaler()
    unsw_nb15[numerical_features] = scaler.fit_transform(unsw_nb15[numerical_features])

    # Apply PCA to reduce numerical features to 15 components
    pca = PCA(n_components=15)
    unsw_nb15_pca = pca.fit_transform(unsw_nb15[numerical_features])
    unsw_nb15_pca_df = pd.DataFrame(unsw_nb15_pca, columns=[f'pca_{i}' for i in range(15)])

    # Combine PCA features with other features
    final_df = pd.concat([unsw_nb15.drop(columns=numerical_features), unsw_nb15_pca_df], axis=1)

    return final_df


# Preprocess the datasets
processed_kddcup_99 = preprocess_kddcup(kddcup_99)
processed_unsw_nb15 = preprocess_unsw_nb15(unsw_nb15)

# Combine the datasets
combined_df = pd.concat([processed_kddcup_99])

# Split data function
def split_data(df, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits + 1)
    splits = list(tscv.split(df))

    train_val_idx, test_idx = splits[0]
    train_idx, val_idx = splits[1]

    train_val_set = df.iloc[train_val_idx]
    test_set = df.iloc[test_idx]
    train_set = df.iloc[train_idx]
    val_set = df.iloc[val_idx]

    return train_set, val_set, test_set

# Split the combined data
train_set, val_set, test_set = split_data(combined_df)

# Prepare data for model
def prepare_data(train_set, val_set, test_set):
    X_train, y_train = train_set.drop(columns=['label']), train_set['label']
    X_val, y_val = val_set.drop(columns=['label']), val_set['label']
    X_test, y_test = test_set.drop(columns=['label']), test_set['label']

    # Reshape data for GRU (samples, time steps, features)
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_set, val_set, test_set)

# Build the Base4-GRU model
def build_base4_gru_model(input_shape):
    model = Sequential()
    
    # First GRU layer
    model.add(GRU(64, input_shape=input_shape, return_sequences=True, dropout=0.01, recurrent_dropout=0.01))
    model.add(LayerNormalization())
    model.add(Dropout(0.01))
    
    # Second GRU layer
    model.add(GRU(64, return_sequences=True, dropout=0.01, recurrent_dropout=0.01))
    model.add(LayerNormalization())
    model.add(Dropout(0.01))
    
    # Third GRU layer
    model.add(GRU(64, return_sequences=True, dropout=0.01, recurrent_dropout=0.01))
    model.add(LayerNormalization())
    model.add(Dropout(0.01))
    
    # Fourth GRU layer
    model.add(GRU(64, return_sequences=False, dropout=0.01, recurrent_dropout=0.01))
    model.add(LayerNormalization())
    model.add(Dropout(0.01))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Compile and train the Base4-GRU model
input_shape = (X_train.shape[1], X_train.shape[2])
base4_gru_model = build_base4_gru_model(input_shape)

base4_gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start_time = time.time()
history_base4_gru = base4_gru_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])
end_time = time.time()
training_time_base4_gru = end_time - start_time

# Evaluate the Base4-GRU model
loss_base4_gru, accuracy_base4_gru = base4_gru_model.evaluate(X_test, y_test)
print(f'Base4-GRU Test Accuracy: {accuracy_base4_gru}, Test Loss: {loss_base4_gru}, Training Time: {training_time_base4_gru} seconds')

# Save the Base4-GRU model
base4_gru_model.save('base4_gru_model.h5')

# Fine-tune the Base4-GRU model
def fine_tune_base4_gru_model(base_model_path, X_train, y_train, X_val, y_val):
    base_model = load_model(base_model_path)
    
    # Here we don't change the architecture, just reload the model and adjust learning rates or continue training.
    base_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    start_time = time.time()
    history = base_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])
    end_time = time.time()
    training_time_fine_tuned = end_time - start_time
    
    return base_model, history, training_time_fine_tuned

fine_tuned_base4_gru_model, history_fine_tuned, training_time_fine_tuned = fine_tune_base4_gru_model('base4_gru_model.h5', X_train, y_train, X_val, y_val)

# Evaluate the Fine-Tuned Base4-GRU model
loss_fine_tuned, accuracy_fine_tuned = fine_tuned_base4_gru_model.evaluate(X_test, y_test)
print(f'Fine-Tuned Base4-GRU Test Accuracy: {accuracy_fine_tuned}, Test Loss: {loss_fine_tuned}, Training Time: {training_time_fine_tuned} seconds')

# Build the TL-3GRU-1 model
def build_tl_3gru_1_model(input_shape, base_model):
    model = Sequential()
    
    # First GRU layer
    model.add(GRU(64, input_shape=input_shape, return_sequences=True, dropout=0.01, recurrent_dropout=0.01))
    model.add(LayerNormalization())
    model.add(Dropout(0.01))
    
    # Second GRU layer
    model.add(GRU(64, return_sequences=True, dropout=0.01, recurrent_dropout=0.01))
    model.add(LayerNormalization())
    model.add(Dropout(0.01))
    
    # Third GRU layer
    model.add(GRU(64, return_sequences=False, dropout=0.01, recurrent_dropout=0.01))
    model.add(LayerNormalization())
    model.add(Dropout(0.01))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Transfer weights from Base4-GRU model
    for i in range(len(model.layers) - 1):  # Exclude the output layer
        model.layers[i].set_weights(base_model.layers[i].get_weights())
        model.layers[i].trainable = False  # Freeze the layers
    
    return model

# Load the Base4-GRU model
base4_gru_model = load_model('base4_gru_model.h5')

# Build the TL-3GRU-1 model
tl_3gru_1_model = build_tl_3gru_1_model(input_shape, base4_gru_model)

# Compile and train the TL-3GRU-1 model
tl_3gru_1_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start_time = time.time()
history_tl_3gru_1 = tl_3gru_1_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])
end_time = time.time()
training_time_tl_3gru_1 = end_time - start_time

# Evaluate the TL-3GRU-1 model
loss_tl_3gru_1, accuracy_tl_3gru_1 = tl_3gru_1_model.evaluate(X_test, y_test)
print(f'TL-3GRU-1 Test Accuracy: {accuracy_tl_3gru_1}, Test Loss: {loss_tl_3gru_1}, Training Time: {training_time_tl_3gru_1} seconds')

# Save the TL-3GRU-1 model
tl_3gru_1_model.save('tl_3gru_1_model.h5')

# Build the TL-4GRU-2 model
def build_tl_4gru_2_model(input_shape, base_model):
    model = Sequential()
    
    # First GRU layer
    model.add(GRU(64, input_shape=input_shape, return_sequences=True, dropout=0.01, recurrent_dropout=0.01, name='gru_1'))
    model.add(LayerNormalization(name='layernorm_1'))
    model.add(Dropout(0.01, name='dropout_1'))
    
    # Second GRU layer
    model.add(GRU(64, return_sequences=True, dropout=0.01, recurrent_dropout=0.01, name='gru_2'))
    model.add(LayerNormalization(name='layernorm_2'))
    model.add(Dropout(0.01, name='dropout_2'))
    
    # Third GRU layer
    model.add(GRU(64, return_sequences=True, dropout=0.01, recurrent_dropout=0.01, name='gru_3'))
    model.add(LayerNormalization(name='layernorm_3'))
    model.add(Dropout(0.01, name='dropout_3'))
    
    # Fourth GRU layer
    model.add(GRU(64, return_sequences=False, dropout=0.01, recurrent_dropout=0.01, name='gru_4'))
    model.add(LayerNormalization(name='layernorm_4'))
    model.add(Dropout(0.01, name='dropout_4'))
    
    # Output layer
    model.add(Dense(1, activation='softmax', name='dense_output'))
    
    # Transfer weights from TL-3GRU-1 model
    for i in range(len(base_model.layers) - 1):  # Exclude the output layer
        if 'gru' in base_model.layers[i].name:
            model.layers[i].set_weights(base_model.layers[i].get_weights())
            model.layers[i].trainable = False  # Freeze the layers
        elif 'layernorm' in base_model.layers[i].name or 'dropout' in base_model.layers[i].name:
            model.layers[i].set_weights(base_model.layers[i].get_weights())
            model.layers[i].trainable = False
    
    return model

# Load the TL-3GRU-1 model
tl_3gru_1_model = load_model('tl_3gru_1_model.h5')

# Build the TL-4GRU-2 model
tl_4gru_2_model = build_tl_4gru_2_model(input_shape, tl_3gru_1_model)

# Compile and train the TL-4GRU-2 model
tl_4gru_2_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start_time = time.time()
history_tl_4gru_2 = tl_4gru_2_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])
end_time = time.time()
training_time_tl_4gru_2 = end_time - start_time

# Evaluate the TL-4GRU-2 model
loss_tl_4gru_2, accuracy_tl_4gru_2 = tl_4gru_2_model.evaluate(X_test, y_test)
print(f'TL-4GRU-2 Test Accuracy: {accuracy_tl_4gru_2}, Test Loss: {loss_tl_4gru_2}, Training Time: {training_time_tl_4gru_2} seconds')

# Save the TL-4GRU-2 model
tl_4gru_2_model.save('tl_4gru_2_model.h5')

# Build the Linear Model
def build_linear_model(input_shape):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape, activation='linear'))
    model.compile(optimizer=SGD(), loss='mse', metrics=['accuracy'])
    return model

# Prepare data for Linear Model
X_train_linear = train_set.drop(columns=['label']).values
y_train_linear = train_set['label'].values
X_val_linear = val_set.drop(columns=['label']).values
y_val_linear = val_set['label'].values
X_test_linear = test_set.drop(columns=['label']).values
y_test_linear = test_set['label'].values

# Build, compile and train the Linear Model
linear_model = build_linear_model((X_train_linear.shape[1],))

start_time = time.time()
linear_model.fit(X_train_linear, y_train_linear, epochs=50, batch_size=64, validation_data=(X_val_linear, y_val_linear), callbacks=[early_stopping])
end_time = time.time()
training_time_linear = end_time - start_time

# Evaluate the Linear Model
loss_linear, accuracy_linear = linear_model.evaluate(X_test_linear, y_test_linear)
print(f'Linear Model Test Accuracy: {accuracy_linear}, Test Loss: {loss_linear}, Training Time: {training_time_linear} seconds')

# Save the Linear model
linear_model.save('linear_model.h5')

# Prepare data for the Wide & Deep Model
def prepare_wide_deep_data(train_set, val_set, test_set):
    # Prepare wide data (linear model input)
    X_train_wide = train_set.drop(columns=['label']).values
    X_val_wide = val_set.drop(columns=['label']).values
    X_test_wide = test_set.drop(columns=['label']).values
    
    # Prepare deep data (GRU model input)
    X_train_deep = np.expand_dims(train_set.drop(columns=['label']), axis=1)
    X_val_deep = np.expand_dims(val_set.drop(columns=['label']), axis=1)
    X_test_deep = np.expand_dims(test_set.drop(columns=['label']), axis=1)
    
    y_train = train_set['label'].values
    y_val = val_set['label'].values
    y_test = test_set['label'].values
    
    return X_train_wide, X_train_deep, y_train, X_val_wide, X_val_deep, y_val, X_test_wide, X_test_deep, y_test

# Prepare data
X_train_wide, X_train_deep, y_train, X_val_wide, X_val_deep, y_val, X_test_wide, X_test_deep, y_test = prepare_wide_deep_data(train_set, val_set, test_set)

# Build the Wide & Deep Model
def build_wide_deep_model(input_shape_wide, input_shape_deep, linear_model, deep_model):
    # Wide component (Linear Model)
    wide_input = Input(shape=(input_shape_wide,))
    wide_output = linear_model(wide_input)
    
    # Deep component (TL-4GRU-2 Model)
    deep_input = Input(shape=input_shape_deep)
    deep_output = deep_model(deep_input)
    
    # Concatenate wide and deep components
    merged_output = Concatenate()([wide_output, deep_output])
    
    # Output layer
    final_output = Dense(1, activation='sigmoid')(merged_output)
    
    # Build the final model
    model = Model(inputs=[wide_input, deep_input], outputs=final_output)
    
    return model

# Define input shapes for wide and deep components
input_shape_wide = X_train_wide.shape[1]
input_shape_deep = (X_train_deep.shape[1], X_train_deep.shape[2])

# Build the Wide & Deep Model
wide_deep_model = build_wide_deep_model(input_shape_wide, input_shape_deep, linear_model, tl_4gru_2_model)

# Compile the Wide & Deep Model
wide_deep_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the Wide & Deep Model
start_time = time.time()
history_wide_deep = wide_deep_model.fit([X_train_wide, X_train_deep], y_train, epochs=50, batch_size=64, validation_data=([X_val_wide, X_val_deep], y_val), callbacks=[early_stopping])
end_time = time.time()
training_time_wide_deep = end_time - start_time

# Evaluate the Wide & Deep Model
evaluation_results = wide_deep_model.evaluate([X_test_wide, X_test_deep], y_test)
loss_wide_deep, accuracy_wide_deep = evaluation_results[0], evaluation_results[1]
print(f'Wide & Deep TL Stacked GRU Model Test Accuracy: {accuracy_wide_deep}, Test Loss: {loss_wide_deep}, Training Time: {training_time_wide_deep} seconds')

# Print the model summary to check the metrics
wide_deep_model.summary()

# Check the metrics used in the model
print(wide_deep_model.metrics_names)

# Save the Wide & Deep Model
wide_deep_model.save('wide_deep_tl_stacked_gru_model.h5')
