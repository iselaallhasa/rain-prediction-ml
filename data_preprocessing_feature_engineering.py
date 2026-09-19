
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('weatherAUS.csv')

print(df.head())

print(df.shape)
print(df.columns)
print(df.info())

print(df.describe())

plt.figure(figsize=(6,4))
sns.countplot(x='RainTomorrow', data=df)
plt.title('Rain Tomorrow Distribution')
plt.show()


missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (df.isnull().sum()/len(df))*100

missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

print(missing_df)

plt.figure(figsize=(14,8))
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing Values Heatmap')
plt.show()


numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

for col in numerical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(x='Humidity3pm', y='Pressure3pm', hue='RainTomorrow', data=df)
plt.title('Dense Areas Scatter Plot')
plt.show()


correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(16,12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


categorical_features = df.select_dtypes(include=['object']).columns

for col in categorical_features:
    print(df[col].value_counts())
    print('-'*50)


num_imputer = SimpleImputer(strategy='median')

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')

categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print(df.isnull().sum())


print('Duplicates Before:', df.duplicated().sum())

df = df.drop_duplicates()

print('Duplicates After:', df.duplicated().sum())

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])



# FEATURE ENGINEERING PART 

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


if 'Rainfall' in df.columns:
    df['Rainfall_Log'] = np.log1p(df['Rainfall'])

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    sns.histplot(df['Rainfall'], kde=True)
    plt.title('Original Rainfall')

    plt.subplot(1,2,2)
    sns.histplot(df['Rainfall_Log'], kde=True)
    plt.title('Log Transformed Rainfall')

    plt.show()


df['RainTodayBinary'] = df['RainToday'].map({'Yes':1, 'No':0})


label_encoders = {}

for col in categorical_features:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    label_encoders[col] = encoder

print("Categorical variables encoded successfully!")

df['TempDifference'] = df['MaxTemp'] - df['MinTemp']
df['HumidityDifference'] = df['Humidity3pm'] - df['Humidity9am']
df['PressureDifference'] = df['Pressure3pm'] - df['Pressure9am']

df['WindSpeedInteraction'] = df['WindSpeed9am'] * df['WindSpeed3pm']

df['WindGustRatio'] = df['WindGustSpeed'] / (df['WindSpeed3pm'] + 1)

df['AvgTemp'] = (df['MinTemp'] + df['MaxTemp']) / 2
df['AvgHumidity'] = (df['Humidity9am'] + df['Humidity3pm']) / 2
df['AvgPressure'] = (df['Pressure9am'] + df['Pressure3pm']) / 2

df['TempHumidityIndex'] = df['AvgTemp'] * df['AvgHumidity']
df['PressureHumidityInteraction'] = df['AvgPressure'] * df['AvgHumidity']


scaler = StandardScaler()

feature_columns = df.drop('RainTomorrow', axis=1).columns

scaled_features = scaler.fit_transform(df[feature_columns])

scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)

scaled_df['RainTomorrow'] = df['RainTomorrow']


print(scaled_df.head())


plt.figure(figsize=(18,14))
sns.heatmap(scaled_df.corr(), cmap='viridis')
plt.title('Final Correlation Matrix')
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(
    x='Humidity3pm',
    y='Pressure3pm',
    hue='RainTomorrow',
    data=scaled_df
)

plt.title('Dense Area Scatter Plot')
plt.show()


print(scaled_df.info())
print(scaled_df.head())


selected_features = [
    'MinTemp',
    'MaxTemp',
    'Humidity3pm',
    'Pressure3pm',
    'RainTomorrow'
]

sns.pairplot(scaled_df[selected_features], hue='RainTomorrow')
plt.show()

scaled_df.to_csv('weather_cleaned_preprocessed.csv', index=False)

print('Cleaned dataset saved successfully!')
print(df.isnull().sum())