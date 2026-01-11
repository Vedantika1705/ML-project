import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset (your path is already correct)
df = pd.read_csv('artifacts/data.csv')  

# Create an images folder in the project root if it doesn't exist
images_path = os.path.join(os.getcwd(), 'images')  # This uses current working directory
os.makedirs(images_path, exist_ok=True)

# 1. Missing Values Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.savefig(os.path.join(images_path, 'missing_values.png'))
plt.close()

# 2. Correlation Matrix
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.savefig(os.path.join(images_path, 'correlation_matrix.png'))
plt.close()

# 3. Feature Distributions
num_cols = df.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(images_path, f'{col}_distribution.png'))
    plt.close()

print(f"EDA visualizations saved to: {images_path}")
