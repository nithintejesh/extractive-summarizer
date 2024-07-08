# extractive-summarizer

This Python project fetches web content from a given URL and generates a summary using TF-IDF (Term Frequency-Inverse Document Frequency) scoring.

## Features
- Fetches and processes web content from the provided URL.
- Tokenizes text into sentences and words.
- Calculates TF-IDF matrix for the text.
- Scores sentences based on their TF-IDF values.
- Generates a summary by selecting sentences with scores above a certain threshold.

## Requirements
- `beautifulsoup4`
- `nltk`
- `requests`

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/articlesummarizer.git
    cd articlesummarizer
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download NLTK data**:
    Uncomment and run the following lines in your code or run them separately:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    ```
## Running the Script

After completing the setup steps, you can run the summarizer script:

### Command Line

Run the following command in your terminal:

```sh
python summarizer.py

