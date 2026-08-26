from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(max_features=5000, min_df=1):
    """
    Initializes a TF-IDF Vectorizer with unigram and bigram ranges
    and returns sparse CSR matrices[cite: 1].
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df
    )