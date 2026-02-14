# AI-ML-Internship-Task2-Titanic-EDA
# ==============================
# AI & ML Internship - Task 2
# Exploratory Data Analysis (EDA)
# Titanic Dataset
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 2. Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 Rows:")
print(df.head())

# 3. Dataset Information
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# 4. Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Age Distribution
plt.figure()
df['Age'].hist()
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 6. Fare Distribution
plt.figure()
df['Fare'].hist()
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# 7. Fare Boxplot (Outliers)
plt.figure()
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()

# 8. Survival by Gender
plt.figure()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# 9. Survival by Passenger Class
plt.figure()
sns.countplot(x='Pclass', hue='Survived',
