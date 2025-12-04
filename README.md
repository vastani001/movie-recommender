# 🎬 Movie Recommendation System  

A content-based movie recommendation system built using Python, TF-IDF vectorization, and cosine similarity.  
Given a movie title, the system recommends the top-10 most similar movies based on plot, genres, and keywords.

---

## 🚀 Features  
- Uses **TF-IDF vectorization** on movie descriptions  
- Computes **cosine similarity** across all movies  
- Recommends **Top-10 similar movies** for any title  
- Works with any dataset containing:  
  - `title`  
  - `overview`  
  - `genres` (optional)  
  - `keywords` (optional)  
- Clean **modular code** (training + recommendation separated)  
- Fast, lightweight, and easy to extend  

---

## 📂 Project Structure  

movie-recommender/
├─ data/
│ └─ movies.csv
├─ src/
│ └─ movie_recommender.py
├─ main.py
├─ requirements.txt
└─ README.md



---

## 🧠 How It Works  

### **1️⃣ Build Text Corpus**  
We combine movie metadata into a single text for each movie:  
corpus = overview + genres + keywords



### **2️⃣ TF-IDF Vectorization**  
Convert the text into numeric feature vectors using:

TfidfVectorizer(stop_words="english", max_features=5000)



### **3️⃣ Cosine Similarity**  
Compute similarity scores between every pair of movies.

### **4️⃣ Recommend Movies**  
Given a movie title:  
- Find its index  
- Lookup similarity scores  
- Sort movies by similarity  
- Return the top-10 most similar  

---

## ▶️ Run the Project  

### **Install dependencies:**
pip install -r requirements.txt



### **Run the CLI tool:**
python main.py



You will see:

Enter a movie title to get recommendations (or 'q' to quit):

Avatar



Output:

Top recommendations similar to 'Avatar':

John Carter

Guardians of the Galaxy

Star Wars: The Force Awakens
...


---

## 📊 Dataset  

You can use any movie dataset containing:

- `title`  
- `overview`  
- `genres` (optional)  
- `keywords` (optional)  

Place your dataset here:

data/movies.csv


Recommended dataset: **TMDB 5000 Movie Dataset (Kaggle).**

---

## 📌 Requirements  

pandas
numpy
scikit-learn


---

## 🌱 Future Improvements  
- Streamlit Web App UI  
- Fuzzy search for partial movie names  
- Combine popularity/ratings for smarter ranking  
- Hybrid model (content-based + collaborative filtering)  
- Add poster / image previews  

---

## 👤 Author  
**Vastani Yash**  
GitHub: https://github.com/vastani001
