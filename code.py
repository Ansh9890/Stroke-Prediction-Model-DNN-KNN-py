import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Input

df = pd.read_csv("test.csv")
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)

df['bmi'].fillna(df['bmi'].mean(), inplace=True)
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.36, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=5/9, random_state=42, stratify=y_temp)

print("Deep Learning Model (DNN):")
dnn_params = {
    'input_shape': X_train.shape[1],
    'hidden_layers': [128, 96, 64],
    'activation': 'selu',
    'output_size': 3,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0005),
    'batch_size': 200,
    'epochs': 100,
    'early_stop_patience': 3
}

model = Sequential()
model.add(Input(shape=(dnn_params['input_shape'],)))
for units in dnn_params['hidden_layers']:
    model.add(Dense(units, activation=dnn_params['activation']))
    model.add(BatchNormalization())
model.add(Dense(dnn_params['output_size'], activation='softmax'))
model.compile(optimizer=dnn_params['optimizer'],
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=dnn_params['early_stop_patience'],
                                                  restore_best_weights=True,
                                                  verbose=0)
