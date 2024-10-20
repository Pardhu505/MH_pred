import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
file_path = r'C:\Users\pardh\Downloads\2009,2014,2019AE.xlsx'  # Update this path
data = pd.read_excel(file_path)

# Filter data to only include winning candidates (Position == 1)
winning_candidates = data[data['Position'] == 1]

# Encode categorical variables: Party, Constituency Name
label_encoder_party = LabelEncoder()
label_encoder_constituency = LabelEncoder()

winning_candidates['Party_encoded'] = label_encoder_party.fit_transform(winning_candidates['Party'])
winning_candidates['Constituency_encoded'] = label_encoder_constituency.fit_transform(winning_candidates['Constituency Name'])

# Select features: Vote Share Percentage, Margin Percentage, Constituency, and Year
X = winning_candidates[['Vote Share Percentage', 'Margin Percentage', 'Constituency_encoded', 'Year']]
y = winning_candidates['Party_encoded']

# Split the data into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the label encoders
with open('label_encoder_party.pkl', 'wb') as file:
    pickle.dump(label_encoder_party, file)
    
with open('label_encoder_constituency.pkl', 'wb') as file:
    pickle.dump(label_encoder_constituency, file)
