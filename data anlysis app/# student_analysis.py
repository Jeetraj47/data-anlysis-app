# student_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# --- Basic Exploration ---
print("First 5 rows:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())
# --- Data Cleaning ---

# Rename columns for easier access
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# --- Analysis: Score Distributions ---
plt.figure(figsize=(10, 5))
sns.histplot(df['math_score'], kde=True, color='skyblue')
plt.title("Math Score Distribution")
plt.show()

# --- Average Scores by Gender ---
gender_avg = df.groupby('gender')[['math_score', 'reading_score', 'writing_score']].mean()
print("\nAverage Scores by Gender:\n", gender_avg)

gender_avg.plot(kind='bar', figsize=(8, 5), title="Average Scores by Gender")
plt.ylabel("Score")
plt.show()

# --- Test Preparation Effect ---
plt.figure(figsize=(8, 5))
sns.boxplot(x='test_preparation_course', y='math_score', data=df)
plt.title("Math Score vs Test Preparation")
plt.show()

# --- Correlation Matrix ---
corr = df[['math_score', 'reading_score', 'writing_score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Score Correlation Matrix")
plt.show()

