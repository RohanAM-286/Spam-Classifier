import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure nltk data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Email / SMS Spam Classifier")

st.sidebar.title("About")
st.sidebar.info("This app classifies whether a message is Spam or Not using a trained ML model and TF-IDF vectorization.")
st.sidebar.info("Built by Rohan")

# Sample messages
sample_messages = [
    "Congratulations! You've won a free iPhone. Click to claim.",
    "Hey, are we still meeting at 5?",
    "URGENT: Your account has been suspended. Log in to verify.",
    "This is your OTP: 2354. Please do not share it with anyone.",
    "Limited offer only for you. Win Rs. 1 Lakh now!"
]

selected = st.selectbox("👇 Choose a sample message (optional)", sample_messages)
input_sms = st.text_area("Or type your message below:", selected)

# Predict Button
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        st.markdown("**Transformed Text (Preprocessing Phase):**")
        st.code(transformed_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]
        confidence = round(proba[result] * 100, 2)

        # Display Result
        if result == 1:
            st.error(f"🛑 **Spam** (Confidence: {confidence}%)")
        else:
            st.success(f"✅ **Not Spam** (Confidence: {confidence}%)")
