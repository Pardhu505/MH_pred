import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


model = pickle.load(open('model.pkl', 'rb'))
label_encoder_party = pickle.load(open('label_encoder_party.pkl', 'rb'))
label_encoder_constituency = pickle.load(open('label_encoder_constituency.pkl', 'rb'))

# Simulated data for dropdowns (replace with real data if needed)
constituencies = ['VANDRE WEST', 'BHIWANDI WEST', 'SILLOD', 'Others']

# Streamlit Title and Input Form
st.title("Maharashtra Election Seat Predictor")

# Dropdown for Constituency Name
constituency_name = st.selectbox('Enter Constituency Name', constituencies)

# Placeholder values for vote share and margin, can be dynamic if connected to real data
vote_share = st.slider('Enter Vote Share Percentage', 0, 100, 50)  # Slider for vote share percentage
margin = st.slider('Enter Margin Percentage', 0, 50, 5)            # Slider for margin percentage
year = st.selectbox('Select Year', [2009, 2014, 2019])              # Year of election

# 'Analyze' button
if st.button("Analyze"):
    # Encode the constituency name
    constituency_encoded = label_encoder_constituency.transform([constituency_name])[0]
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Vote Share Percentage': [vote_share],
        'Margin Percentage': [margin],
        'Constituency_encoded': [constituency_encoded],
        'Year': [year]
    })

    # Predict probabilities for each party
    pred_probs = model.predict_proba(input_data)[0]
    parties = label_encoder_party.inverse_transform(np.argsort(pred_probs)[::-1])  # Sorted parties by probability
    sorted_probs = np.sort(pred_probs)[::-1]

    # Display party-wise probabilities
    st.subheader("Party-wise Probabilities")
    for party, prob in zip(parties, sorted_probs):
        st.write(f"{party}: {prob:.2f}")

    # Plot the probabilities
    st.subheader("Probability Distribution")
    plt.figure(figsize=(10, 6))
    plt.bar(parties, sorted_probs, color='skyblue')
    plt.title("Party-wise Probability Distribution")
    plt.xlabel("Party")
    plt.ylabel("Probability")
    st.pyplot(plt)

    # Detailed Explanation
    st.subheader("Detailed Analysis")
    st.write("""
    The prediction is based on historical election data, including vote share percentage, margin percentage, and the constituency's historical performance.

    - **Vote Share Percentage**: The higher the vote share, the more likely a party is to win.
    - **Margin Percentage**: Larger margins tend to indicate stronger dominance in the constituency.
    - **Year Trends**: Historical trends from the constituency across different election years are considered.
    """)

    # Example of a trend plot (simulated here)
    st.subheader(f"Historical Vote Share Trend for {constituency_name}")
    years = [2009, 2014, 2019]
    historical_vote_share = [45, 50, 52]  # Example data (replace with real data)
    plt.figure(figsize=(10, 6))
    plt.plot(years, historical_vote_share, marker='o')
    plt.title(f"Vote Share Trend for {constituency_name}")
    plt.xlabel("Year")
    plt.ylabel("Vote Share (%)")
    st.pyplot(plt)
