import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('ER Wait Time Dataset.csv', parse_dates=['Visit Date'])

agg = df.groupby(['Day of Week', 'Season', 'Time of Day']).size().reset_index(name='num_patients')
agg.to_csv('num_patients_predictions.csv', index=False)

le_day = LabelEncoder()
agg['Day_encoded'] = le_day.fit_transform(agg['Day of Week'])

le_season = LabelEncoder()
agg['Season_encoded'] = le_season.fit_transform(agg['Season'])

le_time = LabelEncoder()
agg['Time_encoded'] = le_time.fit_transform(agg['Time of Day'])

X = agg[['Day_encoded', 'Season_encoded', 'Time_encoded']]
y = agg['num_patients']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Train R2 score: {model.score(X_train, y_train):.3f}")
print(f"Test R2 score: {model.score(X_test, y_test):.3f}")

joblib.dump(model, 'model_2.pkl')
joblib.dump(le_day, 'label_encoder_day.pkl')
joblib.dump(le_season, 'label_encoder_season.pkl')
joblib.dump(le_time, 'label_encoder_time.pkl')

print("Model i label encoderi su spremljeni.")
