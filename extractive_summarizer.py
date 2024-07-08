import requests
import re
import bs4 as bs
import nltk
import sys
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Uncomment and run this if executing for the first time
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess the input text by removing square brackets, extra spaces, and converting to lowercase.
    """
    text = re.sub(r'\[[0-9]*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# Function to fetch web content from URL
def fetch_web_content(url):
    """
    Fetch the web content from the provided URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        parsed_article = bs.BeautifulSoup(response.text, 'html.parser')
        paragraphs = parsed_article.find_all('p')
        article_text = ''.join([p.text for p in paragraphs])
        return preprocess_text(article_text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing the URL content: {e}")
        sys.exit(1)

# Function to tokenize text into sentences
def tokenize_sentences(text):
    """
    Tokenize the input text into sentences.
    """
    return sent_tokenize(text)

# Function to tokenize words and remove stopwords
def tokenize_words(sentence, stop_words):
    """
    Tokenize the sentence into words, remove stopwords, and lemmatize the words.
    """
    words = word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word.lower() not in stop_words]

# Function to calculate TF-IDF matrix
def tf_idf_matrix(sentences, stop_words):
    """
    Calculate the TF-IDF matrix for the given sentences.
    """
    # TF Calculation
    tf_matrix = {}
    for i, sent in enumerate(sentences):
        tf_table = {}
        words = tokenize_words(sent, stop_words)
        for word in words:
            tf_table[word] = tf_table.get(word, 0) + 1
        total_words = len(words)
        for word in tf_table:
            tf_table[word] = tf_table[word] / total_words
        tf_matrix[i] = tf_table

    # IDF Calculation
    idf_matrix = {}
    total_sentences = len(sentences)
    for i in range(total_sentences):
        for word in tf_matrix[i]:
            idf_matrix[word] = idf_matrix.get(word, 0) + 1
    for word in idf_matrix:
        idf_matrix[word] = math.log(total_sentences / float(idf_matrix[word]))

    # TF-IDF Calculation
    tf_idf = {i: {word: tf_matrix[i][word] * idf_matrix[word] for word in tf_matrix[i]} for i in tf_matrix}

    return tf_idf

# Function to score sentences based on TF-IDF matrix
def score_sentences(sentences, tf_idf):
    """
    Score sentences based on their TF-IDF values.
    """
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        score = sum(tf_idf[i].values())
        sentence_scores[sent] = score / len(sent.split())
    return sentence_scores

# Function to generate summary
def generate_summary(sentences, sentence_scores, threshold):
    """
    Generate a summary based on the sentence scores and threshold.
    """
    summary = ""
    for sent in sentences:
        if sentence_scores.get(sent, 0) >= threshold:
            summary += " " + sent
    return summary

def main():
    url = input("Enter URL of the article page: ")
    article_text = fetch_web_content(url)
    sentences = tokenize_sentences(article_text)
    stop_words = set(stopwords.words("english"))
    tf_idf = tf_idf_matrix(sentences, stop_words)
    sentence_scores = score_sentences(sentences, tf_idf)
    threshold = sum(sentence_scores.values()) / len(sentence_scores)
    summary = generate_summary(sentences, sentence_scores, 1.3 * threshold)

    print("\n\n")
    print("-" * 20, "Summary", "-" * 20)
    print()
    print(summary)
    print()
    print("-" * 40)
    print("Total words in original article =", len(word_tokenize(article_text)))
    print("Total words in summarized article =", len(word_tokenize(summary)))

if __name__ == "__main__":
    main()
