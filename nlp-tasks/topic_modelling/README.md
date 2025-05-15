# Cryptocurrency Topic Modeling

This project performs **topic modeling** on Reddit posts related to cryptocurrency and finance. The goal is to extract meaningful topics from the data using **Latent Dirichlet Allocation (LDA)** and **Non-Negative Matrix Factorization (NMF)**.

---

## Features

- **Data Collection**: Fetches Reddit posts using the `PRAW` library.
- **Data Preprocessing**: Cleans and prepares text data (removes duplicates, missing values, punctuation, stopwords, and performs lemmatization).
- **Exploratory Analysis**: Includes text length analysis and word cloud generation.
- **Topic Modeling**: Implements LDA and NMF for topic extraction.
- **Visualization**: Generates bar charts, word clouds, hierarchical clustering, and word similarity heatmaps for topics.
- **Model Comparison**: Compares LDA and NMF based on topic coherence and interpretability.

---

## Installation

To run this project, install the required dependencies:

```bash
pip install pyLDAvis gensim nltk praw pandas numpy seaborn matplotlib scikit-learn wordcloud
```

## Usage

1. Set Up Reddit API Credentials:
   Replace the client_id, client_secret, and user_agent in the praw.Reddit setup with your Reddit API credentials.

2. Run the Notebook:
   Open the cryptocurrency_topic_modelling.ipynb file in Jupyter Notebook or VS Code.
   Execute the cells sequentially to collect data, preprocess it, and perform topic modeling.

## Workflow

1. Data Collection:

   - Fetches Reddit posts from specified subreddits using keywords like cryptocurrency, blockchain, bitcoin, etc.
   - Saves the data to a CSV file for further processing.

2. Data Preprocessing:

   - Removes duplicates and missing values.
   - Cleans text by removing punctuation, converting to lowercase, removing stopwords, and lemmatizing words.

3. Exploratory Analysis:

   - Analyzes text lengths and generates a word cloud to visualize the most frequent words.

4. Feature Extraction:

   - Extracts features using CountVectorizer (TF) and TfidfVectorizer (TF-IDF).

5. Topic Modeling:

   - Applies LDA and NMF to extract topics.
   - Visualizes topics using bar charts, word clouds, hierarchical clustering, and heatmaps.

6. Model Comparison:

   - Compares LDA and NMF based on topic coherence and interpretability.

## Results

- NMF:

  - Produced more coherent and interpretable topics.
  - Worked well with the TF-IDF representation, emphasizing meaningful terms.

- LDA:
  - Generated interpretable topics but showed some overlap between topics.
  - Better suited for probabilistic topic distributions.

## Conclusion

- NMF outperformed LDA in this project, providing clearer and more distinct topics.
- Future improvements could include:
  - Hyperparameter tuning for both models.
  - Using domain-specific embeddings like Word2Vec or GloVe.
  - Expanding the dataset with more subreddits or time periods.

## Visualizations

- Word Clouds: Highlight the most frequent words in each topic.
- Bar Charts: Show the top words for each topic.
- Hierarchical Clustering: Displays relationships between topics.
- Heatmaps: Visualize word similarity within topics.

## Acknowledgments

- PRAW for Reddit API integration.
- Gensim for LDA modeling.
- Scikit-learn for NMF and feature extraction.
- NLTK for text preprocessing.
- PyLDAvis for interactive topic visualization.
