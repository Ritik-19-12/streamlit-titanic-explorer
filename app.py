
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model
try:
    model = joblib.load('model/titanic_model.pkl')
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload 'titanic_model.pkl' in the same directory.")
    st.stop()

st.title("🚢 Titanic Survival Prediction")
st.write("Enter the details of a passenger to predict if they would survive the Titanic disaster.")

# Sidebar for user input
st.sidebar.header("Passenger Features")

def user_input():
    Pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    Sex = st.sidebar.selectbox("Sex", ['male', 'female'])
    Age = st.sidebar.slider("Age", 0, 80, 30)
    SibSp = st.sidebar.number_input("No. of Siblings/Spouses Aboard", 0, 10, 0)
    Parch = st.sidebar.number_input("No. of Parents/Children Aboard", 0, 10, 0)
    Fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 32.2)
    Embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
    
    data = {
        'Pclass': Pclass,
        'Sex': 1 if Sex == 'male' else 0,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': 0 if Embarked == 'C' else (1 if Embarked == 'Q' else 2)
    }
    return pd.DataFrame([data])

input_df = user_input()

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"🎉 This passenger **would survive** with a probability of {proba:.2%}")
    else:
        st.error(f"💀 This passenger **would not survive**. Survival chance is {proba:.2%}")

# Dataset insights
st.markdown("---")
st.subheader("📊 Titanic Dataset Insights")

uploaded_file = st.file_uploader("Upload Titanic CSV file for visualizations (optional)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Show first few rows
    st.dataframe(df.head())

    # Visualization: Survival by sex
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Sex', hue='Survived', ax=ax1)
    ax1.set_title("Survival Count by Sex")
    st.pyplot(fig1)

    # Visualization: Survival by class
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=ax2)
    ax2.set_title("Survival Count by Passenger Class")
    st.pyplot(fig2)

    # Age distribution
    fig3, ax3 = plt.subplots()
    sns.histplot(df['Age'].dropna(), bins=30, kde=True, ax=ax3)
    ax3.set_title("Age Distribution")
    st.pyplot(fig3)
else:
    st.info("📎 Upload the Titanic CSV file to see visualizations.")

