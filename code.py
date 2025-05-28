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
