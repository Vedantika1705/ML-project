import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv('artifacts/data.csv')

# Create a target column (example: passed reading score >= 50)
df['target'] = df['reading_score'] >= 50  # True/False

# Select features (drop target and optionally any non-numeric columns)
X = df.drop(['target', 'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'], axis=1)
y = df['target']

# Train RandomForest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Create importance DataFrame
importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)

# Create images folder in project root if not exists
images_path = os.path.join(os.getcwd(), 'images')  # Saves in ML-project/images/
os.makedirs(images_path, exist_ok=True)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=importance, palette='viridis')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(images_path, 'feature_importance.png'))
plt.close()

print(f"Feature importance plot saved to: {images_path}")
