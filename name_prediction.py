import streamlit as st
import nltk

# Load the trained NaiveBayesClassifier from the .pkl file
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = nltk.NaiveBayesClassifier.load(file)
    return model

def predict_gender(name, model):
    features = extract_gender_features(name)  # Assuming you have the same feature extraction method
    return model.classify(features)

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
