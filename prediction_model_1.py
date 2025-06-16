import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv('ER Wait Time Dataset.csv')

features = ['Hospital ID', 'Region', 'Day of Week', 'Season', 'Time of Day', 
            'Urgency Level', 'Nurse-to-Patient Ratio', 'Specialist Availability', 'Facility Size (Beds)']

for col in ['Hospital ID', 'Region', 'Day of Week', 'Season', 'Time of Day', 'Urgency Level']:
    data[col] = data[col].astype('category').cat.codes

X = data[features]
y = data['Total Wait Time (min)']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

with open('model_1.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Model spremljen.")