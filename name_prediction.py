import streamlit as st
import pickle

# Load the trained model from the .pkl file
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_gender(name_input, model):
    gender = model.classify(name_input)
    return gender

def main():
    st.title("Gender Prediction App")
    st.write("Enter a name to predict the gender:")

    # Load the model
    model_path = 'name_prediction.pkl'  # Update with the actual path
    model = load_model(model_path)

    name_input = st.text_input("Name:")
    if name_input:
        predicted_gender = predict_gender(name_input, model)
        st.write(f"Predicted gender for {name_input}: {predicted_gender}")

if __name__ == "__main__":
    main()
