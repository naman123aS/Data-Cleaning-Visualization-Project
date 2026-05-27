
# Data Cleaning and Visualization Project
# Name: Naman Sanadhya

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("------ Data Cleaning & Visualization Project ------")

# Locate dataset (tries script folder first, then current working dir)
dataset_filename = "dataset.csv"
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
dataset_path = os.path.join(script_dir, dataset_filename)
if not os.path.exists(dataset_path):
    if os.path.exists(dataset_filename):
        dataset_path = dataset_filename
    else:
        print(f"Error: '{dataset_filename}' not found. Place the file in {script_dir} or the current working directory.")
        sys.exit(1)

# Loading dataset
df = pd.read_csv(dataset_path)

# Display first few rows
print("\nFirst 5 Records:")
print(df.head())

# Basic information
print("\nDataset Information:")
df.info()

print("\nShape of Dataset:")
print(df.shape)

# Checking missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handling missing values
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_columns:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        if np.isnan(mean_val):
            # if column is all NaN, fill with 0
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(mean_val, inplace=True)

# Filling missing values in object-like columns
object_columns = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

for col in object_columns:
    if df[col].isnull().any():
        modes = df[col].mode()
        fill_val = modes.iloc[0] if not modes.empty else ""
        df[col].fillna(fill_val, inplace=True)

print("\nMissing values handled successfully.")

# Removing duplicate records
duplicates = df.duplicated().sum()
print("\nDuplicate records found:", duplicates)
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed.")
else:
    print("No duplicates to remove.")

# Statistical summary
print("\nStatistical Summary:")
print(df.describe(include='all'))

# Detecting outliers using boxplots
for col in numeric_columns:
    try:
        if df[col].dropna().empty:
            continue
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col], orient='h')
        plt.title("Boxplot of " + str(col))
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Skipping boxplot for {col}: {e}")

# Histogram plots
for col in numeric_columns:
    try:
        if df[col].dropna().empty:
            continue
        plt.figure(figsize=(6,4))
        plt.hist(df[col].dropna(), bins=15)
        plt.title("Histogram of " + str(col))
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Skipping histogram for {col}: {e}")

# Correlation heatmap
try:
    plt.figure(figsize=(10,8))
    correlation = df.corr(numeric_only=True)
    if not correlation.empty:
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        print("No numeric columns available for correlation heatmap.")
except Exception as e:
    print("Could not create correlation heatmap:", e)

print("\nProject Completed Successfully.")

print("\nObservations:")
print("1. Missing values were cleaned.")
print("2. Duplicate records were removed.")
print("3. Outliers were identified using boxplots (if numeric data present).")
print("4. Visualizations were created to understand patterns.")
print("5. Correlation between variables was analyzed (if numeric data present).")
