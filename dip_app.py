import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class DepressionClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DepressionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = DepressionClassifier(input_dim=14)
model.load_state_dict(torch.load("depression_model.pt"))
model.eval()

# --- Define the encoding dictionaries ---
profession_label_encoding = {
    'Teacher': 0, 'Content Writer': 1, 'HR Manager': 2, 'Architect': 3, 'Consultant': 4, 'Pharmacist': 5,
    'Doctor': 6, 'Business Analyst': 7, 'Entrepreneur': 8, 'Chemist': 9, 'Chef': 10, 'Educational Consultant': 11,
    'Data Scientist': 12, 'Researcher': 13, 'Lawyer': 14, 'Customer Support': 15, 'Marketing Manager': 16,
    'Pilot': 17, 'Travel Consultant': 18, 'Plumber': 19, 'Sales Executive': 20, 'Manager': 21, 'Judge': 22,
    'Electrician': 23, 'Financial Analyst': 24, 'Software Engineer': 25, 'Civil Engineer': 26, 'UX/UI Designer': 27,
    'Digital Marketer': 28, 'Accountant': 29, 'Mechanical Engineer': 30, 'Graphic Designer': 31,
    'Research Analyst': 32, 'Investment Banker': 33, 'Family Consultant': 34, 'Medical Doctor': 35
}

city_label_encoding = {
    'Kalyan': 0, 'Patna': 1, 'Vasai-Virar': 2, 'Kolkata': 3, 'Ahmedabad': 4, 'Meerut': 5, 'Ludhiana': 6, 'Pune': 7,
    'Rajkot': 8, 'Visakhapatnam': 9, 'Srinagar': 10, 'Mumbai': 11, 'Indore': 12, 'Agra': 13, 'Surat': 14,
    'Varanasi': 15, 'Vadodara': 16, 'Hyderabad': 17, 'Kanpur': 18, 'Jaipur': 19, 'Thane': 20, 'Lucknow': 21,
    'Nagpur': 22, 'Bangalore': 23, 'Chennai': 24, 'Ghaziabad': 25, 'Delhi': 26, 'Bhopal': 27, 'Faridabad': 28,
    'Nashik': 29, 'Gurgaon': 30
}

degree_label_encoding = {
    'B.Ed': 0, 'B.Arch': 1, 'B.Com': 2, 'BHM': 3, 'BSc': 4, 'B.Pharm': 5, 'BCA': 6, 'M.Ed': 7, 'MCA': 8,
    'BBA': 9, 'MSc': 10, 'LLM': 11, 'LLB': 12, 'M.Tech': 13, 'M.Pharm': 14, 'B.Tech': 15, 'MBA': 16, 'BA': 17,
    'ME': 18, 'MD': 19, 'MHM': 20, 'MBBS': 21, 'BE': 22, 'PhD': 23, 'M.Com': 24, 'MA': 25, 'M.Arch': 26
}

# --- Define a function for prediction ---
def predict_depression(inputs):
    # Scale the input using the loaded scaler
    inputs_scaled = scaler.transform([inputs])
    
    # Convert to torch tensor
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    
    # Get the model prediction (probability of depression)
    with torch.no_grad():
        output = model(inputs_tensor).item()
    
    # Return prediction result (Depressed or Not Depressed)
    if output >= 0.5:
        return "Depression", output
    else:
        return "No Depression", output

# --- Streamlit UI ---
st.title("Depression Prediction")
st.write("Enter the following details to predict depression status:")

# Input fields for the 14 features
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100)
working_prof_or_student = st.selectbox("Working Professional or Student", ["Working", "Student"])
work_pressure = st.slider("Work Pressure (0 - 5)", min_value=0, max_value=5)
job_satisfaction = st.slider("Job Satisfaction (0 - 5)", min_value=0, max_value=5)
dietary_habits = st.selectbox("Dietary Habits", ["Good", "Average", "Poor"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
work_study_hours = st.number_input("Work/Study Hours per Day", min_value=1, max_value=24)
financial_stress = st.slider("Financial Stress (0 - 5)", min_value=0, max_value=5)
family_history_of_mental_illness = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
sleep_category = st.selectbox("Sleep Category", ["Good", "Average", "Poor"])

# Encoding profession, city, and degree
profession = st.selectbox("Profession", list(profession_label_encoding.keys()))
city = st.selectbox("City", list(city_label_encoding.keys()))
degree = st.selectbox("Degree", list(degree_label_encoding.keys()))

# Convert categorical values to numeric encoding if needed
gender_numeric = 0 if gender == "Male" else 1
working_prof_or_student_numeric = 0 if working_prof_or_student == "Working" else 1
dietary_habits_numeric = {"Good": 0, "Average": 1, "Poor": 2}[dietary_habits]
suicidal_thoughts_numeric = 1 if suicidal_thoughts == "Yes" else 0
family_history_of_mental_illness_numeric = 1 if family_history_of_mental_illness == "Yes" else 0
sleep_category_numeric = {"Good": 0, "Average": 1, "Poor": 2}[sleep_category]

# Prepare the input features
user_inputs = [
    gender_numeric,
    age,
    working_prof_or_student_numeric,
    work_pressure,
    job_satisfaction,
    dietary_habits_numeric,
    suicidal_thoughts_numeric,
    work_study_hours,
    financial_stress,
    family_history_of_mental_illness_numeric,
    sleep_category_numeric,
    city_label_encoding[city],  # Encoding city
    profession_label_encoding[profession],  # Encoding profession
    degree_label_encoding[degree]  # Encoding degree
]

# Button to trigger prediction
if st.button("Predict Depression Status"):
    result, prob = predict_depression(user_inputs)
    st.write(f"Prediction: **{result}**")
    st.write(f"Probability of Depression: {prob:.4f}")

    # Optionally, display additional information
    if result == "Depression":
        st.write("We recommend consulting a mental health professional.")
    else:
        st.write("You are showing no signs of depression according to the model.")