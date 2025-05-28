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
