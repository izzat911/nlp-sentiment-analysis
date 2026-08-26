# NLP & Sentiment Analysis Engine

An end-to-end Natural Language Processing (NLP) pipeline built with Python, NLTK, and Scikit-Learn to convert unstructured human language into sparse mathematical arrays and predict text sentiment polarity[cite: 1]. 

Developed as part of **DecodeLabs Project 4: Optional Mastery Phase**[cite: 1].

---

## 🛠️ Key Architectural Features

1. **Custom Stop-Word Filtering**: Preserves critical negation terms (`not`, `no`, `never`, `don't`) by performing set operations against default NLTK stop-word lists to maintain true negative polarity[cite: 1].
2. **POS-Guided Lemmatization**: Utilizes NLTK's `WordNetLemmatizer` paired with Treebank Part-of-Speech (POS) tags (`J`, `V`, `N`, `R`) for accurate morphological reductions[cite: 1].
3. **TF-IDF Vectorization & Feature Bounding**: Translates preprocessed token strings into spatial feature matrices using Unigrams and Bigrams (`ngram_range=(1,2)`) while capping `max_features` to bound vocabulary explosion[cite: 1].
4. **Memory Optimization**: Leverages native SciPy Compressed Sparse Row (CSR) matrix representation to prevent dense array RAM exhaustion ($O(N^3)$ operations)[cite: 1].
5. **Probabilistic Sentiment Inference**: Trains a `MultinomialNB` classifier utilizing Laplace Smoothing ($\alpha = 1.0$) to overcome zero-frequency issues during novel sample inference[cite: 1].

---

## 📁 Repository Structure

```text
nlp-sentiment-analysis/
├── data/
│   └── reviews.csv          # Local dataset storage
├── notebooks/
│   └── project_4_nlp.ipynb  # Interactive Jupyter notebook for experimentation
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Text cleaning & POS-guided lemmatization pipeline
│   └── vectorizer.py        # TF-IDF sparse matrix builder
├── main.py                  # End-to-end execution script
├── requirements.txt         # Dependency declarations
└── README.md                # Project documentation