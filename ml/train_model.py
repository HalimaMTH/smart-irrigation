import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# data بسيطة (temp, humidity, soil → irrigation)
data = {
    "temperature": [30, 25, 35, 20],
    "humidity": [70, 60, 80, 50],
    "soil": [30, 40, 20, 60],
    "irrigation": [1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['temperature', 'humidity', 'soil']]
y = df['irrigation']

model = RandomForestClassifier()
model.fit(X, y)

# save model
joblib.dump(model, "ml/model.pkl")

print("Model créé ✅")