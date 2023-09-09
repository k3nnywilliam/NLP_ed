import gensim
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')

# Sample list of text documents
documents = [
    "Machine learning is a subfield of artificial intelligence.",
    "Natural language processing is important in text analysis.",
    "Deep learning models are used for image recognition.",
    "Topic modeling helps in discovering hidden topics in documents.",
    "Python programming language is commonly used in data science.",
]

# Tokenize and preprocess the text data
stop_words = set(stopwords.words('english'))
punctuations = string.punctuation

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token not in punctuations]
    return tokens

tokenized_documents = [preprocess(doc) for doc in documents]

# Create a dictionary and a document-term matrix
dictionary = corpora.Dictionary(tokenized_documents)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

# Train the LDA model
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Print the topics and their top words
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
