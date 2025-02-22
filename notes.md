## Natural Language Processing

- Machine learning
  - Supervised
    - Classification
      - datasets are inputed
      - a model is created a trained with the dataset to make predictions.
      - most of the datasets are made up of texts.
      - e.g. spam classification.
    - Rgression
  - Unsupervised

## Roadmap

- Text Pre-processing(Cleaning the text)

  - Tokenization : convert paragraphs into sentences, sentences into words
  - Lemmetization
  - Stemming
  - Stop words

- Text Pre-processing(Converting input text to vectors)

  - Bag of Words
  - TFIDF
  - Unigram
  - Bigrams
  - Word Embedding

- Neural Networks (Deep learning technique)

  - word2Vec
  - avgWord2Vec
  - RNN
  - LTSM RNN
  - GRU RNN

- Transformer
- BERT

## Toenization

converting text contents into tokens.

- Corpus : Paragraphs
- Documents : Sentences
- Vocabulary : Unique Words
- Words

Paragraphs => Sentences => Unique Words (Vocabulary)

## One hot encoding

- converting texts to vectors.
- unique vocabulary play important role in one hot encoding. figure out the unique vocabularies first.

        Texts                 O/P

  D1 The food is bad 1
  D2 The food is good 0
  D3 Pizza is amazing 1

Unique words : The food is good bad pizza amazing

- when applying one hot encoding you convert specific words into vector representations of the unique words. example below :

"The" in D1

- vector representation
  The food is good bad pizza amazing
  1 0 0 0 0 0 0

"food" in D1

- vector representation
  The food is good bad pizza amazing
  0 1 0 0 0 0 0

therefore, vector representation of D1

D1 = [[1,0,0,0,0,0,0],
      [0,1,0,0,0,0,0],
      [0,0,1,0,0,0,0],
      [0,0,0,1,0,0,0]]
D1.shape = 4 x 7

D2 = [[1,0,0,0,0,0,0],
      [0,1,0,0,0,0,0],
      [0,0,1,0,0,0,0],
      [0,0,0,0,1,0,0]]
D2.shape = 4 x 7

D3 = [[0,0,0,0,0,1,0],
      [0,0,1,0,0,0,0],
      [0,0,0,0,0,0,1]]
D3.shape = 3 x 7

- disadvantages
  - sparse matrix
  - out of vocabulary
  - semantic meaning still not captured

## bag of words

- step one : dataset available
- step two :
  - change all texts to lowercase
  - apply stopwords
  -
- disadvantages
  - sparse matrix
  - out of vocabulary
  - semantic meaning still not getting captured

## TF - IDF

- Term Frequency is the no of repition of words / no of words in sentence.
- Inverse Document Frequency is the log(no of sentences / no of sentences containing the word)

- advantages

  - intuitive
  - fixed size --> vocab size
  - word importance is getting captured

- disadvantages
  - sparsity still exists
  - OOV

## Word Embedding
  - converts the words in a text to vectors and find words with similar meaning using ttechniques like word2Vec.

## Logistic Regression

- Supervised Learning
  inputs requires a label and has an expected output.

- Sentiment Analysis
  - Tweet : I am happy becuase I am learing NLP
    - Objective : predict wether it is a positive sentiment or negative using logistics regression classifier.
      - Positive : 1
      - Negative : 0
        - Process raw tweet in training set.
        - extract useful features.
        - train logistics regression classifier with minimizing cost.
        - ability to make prediction.

NB : to get the frequency of a word in your vocabulary, you would have to count the number of times it appears.

- word frequency in class
  number of times the word appears on the set of tweets belonging to the class..

- vector of dimension 3
  X(m) = [1(bias),sum(pos(1)),sum(neg(0))]

- Preprocessing Tweets
  - clean texts (tweets)
    - lowercasing
    - removing punctuations, stopwords, handles, urls, applying contractions.. ref: wee2_pipeline_part1
    NB: punctuation can add specific meaning to your nlp task.
  - Stemming/Lemmetization
  - sentence segmentation, word tokenization,
  - stop words, punctuations and digits removal,
  - stemming, Lemmatization,
  - part Of Speech tagging, Code Mixing and translation

<!-- [expression for item in iterable if condition] -->
- expression can be the statement in the for loop
- then takes the condition at the later.

## BERT
  - bidirectional encoder representations from transformer

## Tokenization
  - WhiteSpaceTokenizer (space, tabs, e.t.c)
  - UnicodeScriptTokenizer (words, punctuations, e.t.c)
  - sub word tokenization splits words into smaller unit
  - BERT Tokenizer uses wordPiece
  - SentencePieceTokenizer configurable subwords tokenizer, helps to split OOV words
  - UnicodeCharTokenizer splits words into individual characters useful for text generation
      - helps unders word structure (morphology) e.g unhappy... "un", "happy"
   