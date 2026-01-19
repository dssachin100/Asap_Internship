import streamlit as st
import pandas as pd
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Movie Genre Classifier", layout="centered")
st.title("🎬 Movie Genre Classification App")

# -------------------------------
# Load Data from ZIP
# -------------------------------
@st.cache_data
def load_data_from_zip(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(file_name) as f:
            df = pd.read_csv(
                f,
                sep=":::",
                engine="python",
                names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
            )
    return df

# Change names if needed
train_data = load_data_from_zip(
    "train_data.zip",
    "train_data.txt"
)

# -------------------------------
# Preprocessing
# -------------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000   # keeps memory usage low
)

X_train = tfidf.fit_transform(train_data["DESCRIPTION"])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["GENRE"])

# -------------------------------
# Model Training
# -------------------------------
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# -------------------------------
# User Input
# -------------------------------
st.subheader("📝 Enter Movie Description")

user_input = st.text_area(
    "Movie description:",
    height=150
)

if st.button("🎯 Predict Genre"):
    if user_input.strip() == "":
        st.warning("Please enter a description.")
    else:
        vector = tfidf.transform([user_input])
        prediction = model.predict(vector)
        genre = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🎉 Predicted Genre: **{genre}**")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("ℹ️ App Info")
st.sidebar.write("""
• Dataset loaded from ZIP  
• Algorithm: SVM  
• Vectorizer: TF-IDF  
• Handles large files efficiently  
""")
