import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and label encoders
model = joblib.load("admission_model.pkl")
le_course = joblib.load("label_encoder_course.pkl")
le_country = joblib.load("label_encoder_country.pkl")

# Load dataset and clean column names
data = pd.read_csv("admission_data_universities.csv")
data.columns = data.columns.str.strip()

courses = sorted(data['Preferred Course'].dropna().unique())
countries = sorted(data['Preferred Country'].dropna().unique())

st.title("University Admission Predictor")
st.markdown("""
Welcome to the University Admission Predictor! 🎓  
Use the sidebar to enter your academic profile and preferences, then get your personalized admission chance and university recommendations.
""")

# Sidebar inputs
st.sidebar.header("Academic Profile")
gre = st.sidebar.slider("GRE Score", 260, 340, 300)
toefl = st.sidebar.slider("TOEFL Score", 0, 120, 100)
cgpa = st.sidebar.slider("CGPA out of 10", 0.0, 10.0, 8.0)
sop = st.sidebar.slider("SOP Strength (1-5)", 1.0, 5.0, 3.0)
lor = st.sidebar.slider("LOR Strength (1-5)", 1.0, 5.0, 3.0)
research = st.sidebar.selectbox("Research Experience", [0, 1])

st.sidebar.header("Preferences")
selected_course = st.sidebar.selectbox("Preferred Course", courses)
selected_country = st.sidebar.selectbox("Preferred Country", countries)

if st.sidebar.button("Predict Admission Chance"):
    # Encode categorical selections
    course_encoded = le_course.transform([selected_course])[0]
    country_encoded = le_country.transform([selected_country])[0]

    input_df = pd.DataFrame([[gre, toefl, cgpa, sop, lor, research, course_encoded, country_encoded]],
                            columns=['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR', 'Research',
                                     'Preferred Course', 'Preferred Country'])

    prob = model.predict_proba(input_df)[:, 1][0]
    st.subheader(f"Predicted Chance of Admission: {prob * 100:.2f}%")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    st.subheader("Why this prediction?")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)

    # Recommendations if chance below 75%
    if prob < 0.75:
        recs = data[(data['Preferred Course'] == selected_course) &
                    (data['Preferred Country'] == selected_country) &
                    (data['Chance of Admit'] >= 0.75)][['University Name', 'Chance of Admit ']]

        st.subheader("Recommended Universities:")
        if recs.empty:
            st.write("No recommendations found for your profile.")
        else:
            st.dataframe(recs.style.format({"Chance of Admit": "{:.2f}"}))
