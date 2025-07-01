import streamlit as st
import joblib

model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Spam Message Classifier")

message = st.text_area("Enter a message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        X = vectorizer.transform([message])
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0][1] if hasattr(
            model, "predict_proba") else None
        if prediction == "spam":
            st.error(
                f"This message is likely SPAM. (Confidence: {confidence:.2%})" if confidence is not None else "This message is likely SPAM.")
        else:
            st.success(
                f"This message is likely a HAM. (Confidence: {1-confidence:.2%})" if confidence is not None else "This message is likely a HAM.")
