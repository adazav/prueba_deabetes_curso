from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Diabetes Detection using Machine Learning",
    page_icon="https://images.emojiterra.com/twitter/v13.1/512px/2695.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Diagnostic Measures", "Evaluate Data"],
        icons=["droplet", "droplet-fill"],
        menu_icon="pencil-square"
    )

model = pickle.load(open('models/stacking_rf_oversampling_hp.pkl', 'rb'))

if selected == "Diagnostic Measures":
    st.header('Predict diabetes based on diagnostic measures')
    st.subheader('User Input')

    # Get the feature input from the user
    def get_user_input():
        Pregnancies = st.slider('Pregnancies (Number of Children)', 0, 20, 0)
        Glucose = st.slider('Glucose (mM)', 0, 200, 150)
        Blood_Pressure = st.slider('Blood Pressure (mmHg)', 0, 125, 75)
        Skin_Thickness = st.slider('Skin Thickness (μm)', 0, 100, 25)
        Insulin = st.slider('Insulin (pmol/L)', 0.0, 850.0, 30.0)
        BMI = st.slider('Body Mass Index (BMI)', 0.0, 70.0, 30.0)
        DPF = st.slider('Diabetes Prediction Factor (DPF)', 0.078, 2.500, 0.3725)
        Age = st.slider('Age (years)', 20, 90, 30)

        # Store a dictionary into a variable
        user_data = {
            'Pregnancies (Number of Children)': Pregnancies,
            'Glucose (mM)': Glucose,
            'Blood Pressure (mmHg)': Blood_Pressure,
            'Skin Thickness (μm)': Skin_Thickness,
            'Insulin (pmol/L)': Insulin,
            'Body Mass Index (BMI)': BMI,
            'Diabetes Prediction Factor (DPF)': DPF,
            'Age (years)': Age
        }

        # Transform the data into a dataframe
        features = pd.DataFrame(user_data, index=[0])
        return features


    user_input = get_user_input()

    if st.button("Evaluate"):
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input)
        argmax = np.argmax(probability)
        probability = probability[0]

        # st.subheader('Input Data')
        # st.write(user_input)
        st.subheader('Result')
        classification_result = str(prediction)
        if argmax == 0:
            classification_result = "Not Diabetic"
        else:
            classification_result = "Diabetic"

        st.success(classification_result)
        st.subheader('Accuracy')
        st.success(str((probability[argmax] * 100).round(2)) + "%")

if selected == "Evaluate Data":
    st.header('Evaluate uploaded data from diagnostic measures')
    uploaded_file = st.file_uploader(
        "Upload your data:", type=["csv"]
    )

    if uploaded_file:
        st.subheader('Input Data')
        df = pd.read_csv(uploaded_file, float_precision="round_trip")

        X = df.iloc[:, 0:8].values
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        argmax = np.argmax(probability)
        # probability = probability[0]

        df2 = df[["Pregnancies",
                  "Glucose",
                  "BloodPressure",
                  "SkinThickness",
                  "Insulin",
                  "BMI",
                  "DiabetesPedigreeFunction",
                  "Age"]]

        pred = []
        for i in prediction:
            if i == 0:
                pred.append("Not diabetic")
            else:
                pred.append("Diabetic")

        no_diabetic_accuracy = []
        diabetic_accuracy = []

        for i in probability[:, 0]:
            no_diabetic_accuracy.append(str((i * 100).round(2)) + "%")

        for i in probability[:, 1]:
            diabetic_accuracy.append(str((i * 100).round(2)) + "%")

        df2['Result'] = pred
        df2['No diabetic accuracy'] = no_diabetic_accuracy
        df2['Diabetic accuracy'] = diabetic_accuracy
        st.write(df2)

# Ocultar markdown
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
