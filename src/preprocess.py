import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# --- NLTK Resource Downloads ---
# Downloads required resources quietly to prevent runtime LookupErrors
NLTK_RESOURCES = [
    'stopwords',
    'punkt',
    'punkt_tab',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'wordnet'
]

for resource in NLTK_RESOURCES:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()

# Step 1: Retain negations in stop-words list
# Exclude negation words from default NLTK stop-words to preserve sentiment polarity
default_stopwords = set(stopwords.words('english'))
negations = {'not', 'no', 'nor', 'neither', 'never', 'cannot', "don't", "couldn't", "didn't", "won't"}
custom_stopwords = default_stopwords - negations


def get_wordnet_pos(treebank_tag):
    """
    Maps Treebank POS tags to WordNet POS tags for accurate lemmatization.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text):
    """
    Cleans raw text, tokenizes, filters stop-words (preserving negations),
    and applies POS-guided lemmatization.
    """
    if not isinstance(text, str):
        return ""

    # Character Normalization & Lowercasing: Remove HTML tags, non-alphabetic chars, and lowercase
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Part-of-Speech Tagging
    pos_tags = nltk.pos_tag(tokens)

    # POS-Guided Lemmatization & Stop-word Filtering
    cleaned_tokens = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in custom_stopwords
    ]

    return " ".join(cleaned_tokens)