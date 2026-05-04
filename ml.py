import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Student Marks Predictor", page_icon="👨🏻‍🎓", layout="centered")

st.title("Student Marks Predictor 👨🏻‍🎓👩🏻‍🎓")
st.write("Enter The Number Of Hours Studied 🕒 (1-10) & **Click Predict** To See The Predicted Mark")

# Load The Model
def load_model(model): #inside this we passing the model :slr.pkl
    with open(model,"rb") as f: #open in read format
        #when creating file use dump(), when use it use load
        slr = pickle.load(f) # slr.pkl will be load for you 
    return slr

#if something goes wrong(pickle file not working/any issue/forgot to upload pkl file)
try:
    model = load_model("slr.pkl")
except Exception as e:
    st.error("Your Pickle File Not Found")
    st.exception("Failed To Load The Model :", e)
    st.stop()
    

hours = st.number_input("Hours Studied",
                        min_value = 1.0,
                        max_value = 10.0,
                        value = 4.0,
                        format = "%.1f")

if st.button("Predict"): #if user click this predict button
    try:
        X = np.array([[hours]]) #make this IV into 2d
        predictions = model.predict(X)
        predictions = predictions[0]
        st.success(f" ✅ Predicted Marks: {predictions:.1f}")
        st.write(" ⛔️ Note: This is the ML Model Prediction, **Result May Vary**")
    except Exception as e:
        st.error(f"Prediction Failed : {e}")
        
    