import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("admission_model.pkl")

# Load your full dataset
data = pd.read_csv("admission_data_universities.csv")
data.columns = data.columns.str.strip()  # Clean column names

# Extract dropdown options from data
courses = sorted(data["Preferred Course"].dropna().unique())
countries = sorted(data["Preferred Country"].dropna().unique())

# App title and welcome message
st.title("🎓 University Admission Predictor")

st.markdown("""
Welcome to the **University Admission Predictor**! 🎓  
Use the sidebar to enter your academic profile and preferences, then get your personalized admission chance prediction and university recommendations.
""")

# Sidebar inputs grouped for clarity
st.sidebar.header("Academic Profile")
gre = st.sidebar.slider("GRE Score", 260, 340, 300)
toefl = st.sidebar.slider("TOEFL Score", 0, 120, 100)
cgpa = st.sidebar.slider("CGPA (out of 10)", 0.0, 10.0, 8.0)

st.sidebar.header("Application Strength")
sop = st.sidebar.slider("SOP Strength (1-5)", 1.0, 5.0, 3.0)
lor = st.sidebar.slider("LOR Strength (1-5)", 1.0, 5.0, 3.0)
research = st.sidebar.selectbox("Research Experience", [0, 1])

st.sidebar.header("Preferences")
course = st.sidebar.selectbox("Preferred Course", courses)
country = st.sidebar.selectbox("Preferred Country", countries)

# Predict button with spinner
if st.sidebar.button("Predict Admission Chance"):
    with st.spinner("Calculating your admission chance..."):
        input_data = pd.DataFrame([{
            "GRE Score": gre,
            "TOEFL Score": toefl,
            "CGPA": cgpa,
            "SOP": sop,
            "LOR": lor,
            "Research": research
        }])

        # Predict admission probability
        prediction = model.predict_proba(input_data)[0][1]

    # Show prediction with styled header
    st.markdown(
        f"<h2 style='color: #4CAF50;'>🎯 Predicted Chance of Admission: {prediction:.2%}</h2>", 
        unsafe_allow_html=True
    )

    # Show success or warning based on prediction
    if prediction < 0.75:
        st.warning("Your admission chance is below 75%. Here are some universities you might consider:")
    else:
        st.success("Great! You have a good chance of admission. 🎉")

    # SHAP explanation inside an expander
    with st.expander("🔍 Why this prediction? (SHAP Explanation)"):
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

    # Recommendations inside an expander (only if prediction < 0.75)
    if prediction < 0.75:
        with st.expander("🏫 Recommended Universities"):
            similar = data[
                (data["Preferred Course"].str.lower() == course.lower()) &
                (data["Preferred Country"].str.lower() == country.lower()) &
                (data["Chance of Admit"] >= 0.75)
            ].copy()

            # Calculate CGPA similarity
            similar["CGPA_diff"] = (similar["CGPA"] - cgpa).abs()
            recs = similar.sort_values("CGPA_diff").head(5)

            if not recs.empty:
                st.table(recs[["University Name", "Chance of Admit", "CGPA"]])
            else:
                st.info("No similar universities found. Try broadening your search.")

# Footer
st.markdown("---")


