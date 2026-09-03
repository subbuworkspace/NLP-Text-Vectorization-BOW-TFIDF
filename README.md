# NLP Text Vectorization: Bag of Words & TF-IDF

## 📌 Overview

This project demonstrates the **basic Natural Language Processing (NLP) workflow** for converting human language into numerical data that a machine learning algorithm can understand.

The project uses a sample paragraph about India's development and demonstrates:

* Sentence Tokenization
* Text Cleaning
* Lowercasing
* Stopword Removal
* Lemmatization
* Bag of Words (BoW)
* TF-IDF Vectorization

This project is mainly created as a **learning demonstration of traditional NLP techniques** and how text was converted into numerical representations before modern word and sentence embeddings became widely used.

---

## 🎯 Objective

Computers cannot directly understand human language as text.

For example:

```text
India is a developing nation.
```

Machine learning algorithms need numerical input.

Therefore, NLP techniques convert text into numerical vectors:

```text
Text
 ↓
Preprocessing
 ↓
Vectorization
 ↓
Numerical Representation
 ↓
Machine Learning Model
```

---

## 🧠 NLP Pipeline

The project follows this basic NLP pipeline:

```text
Raw Text
    ↓
Sentence Tokenization
    ↓
Remove Special Characters
    ↓
Convert to Lowercase
    ↓
Remove Stopwords
    ↓
Lemmatization
    ↓
Clean Corpus
    ↓
Bag of Words / TF-IDF
    ↓
Numerical Vectors
```

---

## 📚 Techniques Used

### 1. Sentence Tokenization

The paragraph is divided into individual sentences using NLTK.

```python
sentences = nltk.sent_tokenize(paragraph)
```

Example:

```text
"I have three visions for India."
"It is time we see ourselves as a developed nation."
```

---

### 2. Text Cleaning

Special characters and numbers are removed using Regular Expressions.

```python
review = re.sub('[^a-zA-Z]', ' ', sentences[i])
```

---

### 3. Lowercasing

All words are converted to lowercase.

```python
review = review.lower()
```

For example:

```text
India → india
Development → development
```

This prevents the computer from treating uppercase and lowercase versions as different words.

---

### 4. Stopword Removal

Common words such as:

```text
the
is
and
of
to
in
```

can be removed because they often provide limited information for traditional text classification tasks.

NLTK is used for stopword removal:

```python
from nltk.corpus import stopwords
```

---

### 5. Lemmatization

Lemmatization converts words toward their meaningful base form.

```python
wordnet.lemmatize(word)
```

For example:

```text
cars → car
dogs → dog
```

---

# 🔢 Bag of Words

Bag of Words (BoW) is one of the simplest methods for converting text into numerical data.

It creates a vocabulary of unique words and counts how frequently each word occurs in each document.

Python implementation:

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_bow = cv.fit_transform(corpus).toarray()
```

The result is a **Document-Term Matrix**.

Example:

| Document   | india | development | nation | freedom |
| ---------- | ----: | ----------: | -----: | ------: |
| Document 1 |     1 |           0 |      1 |       2 |
| Document 2 |     2 |           1 |      1 |       0 |

The numbers represent word frequencies.

---

# 📊 TF-IDF

TF-IDF stands for:

**Term Frequency – Inverse Document Frequency**

Instead of simply counting words, TF-IDF calculates how important a word is within a collection of documents.

Python implementation:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

X_tf = tf.fit_transform(corpus).toarray()
```

TF-IDF gives higher importance to words that are relatively important in a document and lower importance to words that appear frequently across many documents.

---

# ⚖️ Bag of Words vs TF-IDF

| Feature                                    | Bag of Words     | TF-IDF           |
| ------------------------------------------ | ---------------- | ---------------- |
| Method                                     | Word counting    | Word weighting   |
| Library                                    | CountVectorizer  | TfidfVectorizer  |
| Output                                     | Numerical vector | Numerical vector |
| Considers word frequency                   | ✅                | ✅                |
| Considers word importance across documents | ❌                | ✅                |
| Understands semantic meaning               | ❌                | ❌                |
| Traditional NLP technique                  | ✅                | ✅                |

---

# 🔍 Is this Vector Embedding?

BoW and TF-IDF produce **vector representations of text**, but they are different from modern semantic embeddings.

Traditional approach:

```text
Text
 ↓
Bag of Words
 ↓
TF-IDF
 ↓
Numerical Vector
```

Modern approach:

```text
Text
 ↓
Word2Vec / GloVe
 ↓
BERT / Transformers
 ↓
Sentence Embedding
 ↓
Semantic Vector
```

BoW and TF-IDF mainly represent **word occurrence and importance**.

Modern embeddings attempt to represent **meaning and relationships between words or sentences**.

---

# 🛠️ Technologies Used

* Python
* NLTK
* Regular Expressions
* NumPy
* Scikit-learn
* Jupyter Notebook / Spyder

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/NLP-Text-Vectorization-BOW-TFIDF.git
```

Move into the project directory:

```bash
cd NLP-Text-Vectorization-BOW-TFIDF
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Download the required NLTK resources:

```python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

# 📄 requirements.txt

```text
nltk
scikit-learn
numpy
jupyter
```

---

# 🚀 How to Run

### Option 1: Jupyter Notebook

Open:

```text
notebooks/NLP_BOW_TFIDF.ipynb
```

Run the cells sequentially.

### Option 2: Python

Run:

```bash
python src/nlp_vectorization.py
```

---

# 📁 Project Structure

```text
NLP-Text-Vectorization-BOW-TFIDF/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── NLP_BOW_TFIDF.ipynb
│
├── src/
│   └── nlp_vectorization.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🎓 Learning Outcome :

* What NLP is
* Why text preprocessing is required
* Sentence tokenization
* Stopword removal
* Lemmatization
* What Bag of Words means
* What a Document-Term Matrix is
* What TF-IDF means
* How text is converted into numerical vectors
* The difference between traditional vectorization and modern embeddings

---

# 🔮 Next Steps

This project represents the **traditional NLP approach**.

The next stage of learning is:

```text
Bag of Words
      ↓
TF-IDF
      ↓
Word2Vec
      ↓
GloVe
      ↓
Word Embeddings
      ↓
RNN / LSTM
      ↓
Attention
      ↓
Transformers
      ↓
BERT
      ↓
Sentence Embeddings
      ↓
LLMs
      ↓
Generative AI
```

This repository there fore serves as a foundation for learning **modern NLP, embeddings, Transformers, and Generative AI**.

---

## 👨‍💻 Author

**Subrata Mondal**

Learning and building projects in:

* Data Science
* Machine Learning
* Natural Language Processing
* Artificial Intelligence
* Generative AI

---

## ⭐ If you find this project useful

Feel free to ⭐ star the repository and use it for your own NLP learning journey.
