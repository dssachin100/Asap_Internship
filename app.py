import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Movie Genre Classifier", layout="centered")
st.title("🎬 Movie Genre Classification App")
st.write("Predict the **genre of a movie** based on its description using ML (TF-IDF + SVM).")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    train_data = pd.read_csv(
        "train_data.txt",
        sep=":::",
        engine="python",
        names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
    )
    return train_data

train_data = load_data()

# -------------------------------
# Preprocessing
# -------------------------------
tfidf = TfidfVectorizer(stop_words="english")
X_train = tfidf.fit_transform(train_data["DESCRIPTION"])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["GENRE"])

# -------------------------------
# Train Model
# -------------------------------
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# -------------------------------
# User Input
# -------------------------------
st.subheader("📝 Enter Movie Description")
user_input = st.text_area(
    "Type or paste the movie description here:",
    height=150
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🎯 Predict Genre"):
    if user_input.strip() == "":
        st.warning("Please enter a movie description.")
    else:
        input_vector = tfidf.transform([user_input])
        prediction = model.predict(input_vector)
        genre = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🎉 Predicted Genre: **{genre}**")

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
- **Algorithm:** Support Vector Machine (SVM)
- **Text Vectorization:** TF-IDF
- **Task:** Movie Genre Classification
""")
