from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd

# Muat dataset
data = pd.read_csv("salaries_cyber.csv")
experience_mapping = {'EN': 1, 'MI': 3, 'SE': 7, 'EX': 10}
data['years_experience'] = data['experience_level'].map(experience_mapping)

# Data dan Target
X = data[['years_experience']]
y = data['salary_in_usd']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Simpan Model
with open('CybSec_Salaries.sav', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil dilatih dan disimpan!")
