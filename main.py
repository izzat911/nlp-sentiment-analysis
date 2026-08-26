import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

from src.preprocess import preprocess_text
from src.vectorizer import build_vectorizer

def load_data():
    csv_path = 'data/reviews.csv'
    
    # Check if user has populated reviews.csv
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        # Assumes CSV has 'text' and 'label' columns (adjust names if needed)
        text_col = 'text' if 'text' in df.columns else 'review'
        label_col = 'label' if 'label' in df.columns else 'sentiment'
        return df[text_col], df[label_col]
    
    print("No CSV found in data/reviews.csv. Using expanded sample dataset...")
    data = {
        'review': [
            # Negative (0)
            "This product is absolutely terrible and broken!!! <br>",
            "I am not happy with the service provided.",
            "Worst purchase ever. Completely useless.",
            "Horrible quality, stopped working immediately.",
            "Waste of money, do not buy this item.",
            "Extremely disappointed with the poor build quality.",
            "The customer service was unresponsive and rude.",
            "Defective item, broke on the very first day.",
            "Not worth the price. Low quality plastic.",
            "Arrived damaged and packaging was ruined.",
            "I regret buying this. Terrible experience overall.",
            "Fails to work as advertised. Very frustrating.",
            
            # Positive (1)
            "Outstanding performance, totally worth the money!",
            "It was not bad, actually quite good and functioning well.",
            "Super fast delivery and brilliant quality overall.",
            "Fantastic product! Very satisfied with my purchase.",
            "Highly recommended, works like a charm!",
            "Great experience! The product exceeded all my expectations.",
            "Exceeded my expectations, wonderful experience.",
            "Love it! Excellent design and top-notch materials.",
            "Impressed with the build quality and smooth delivery.",
            "Works perfectly out of the box, zero issues.",
            "Best purchase I have made this year!",
            "Five stars! Really happy with the customer support."
        ],
        'sentiment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    return df['review'], df['sentiment']

def main():
    texts, labels = load_data()

    print("Preprocessing text (character normalization, stop-word filtering, POS-lemmatization)...")[cite: 1]
    cleaned_texts = texts.apply(preprocess_text)

    print("Vectorizing text via TF-IDF (CSR Sparse Matrix)...")[cite: 1]
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(cleaned_texts)
    y = labels

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train Multinomial Naive Bayes with Laplace Smoothing (alpha=1.0)[cite: 1]
    print("Training Naive Bayes classifier...")[cite: 1]
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    print("\n================ MODEL EVALUATION ================")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))

if __name__ == '__main__':
    main()