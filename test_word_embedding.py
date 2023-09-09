from gensim.models import Word2Vec
model = Word2Vec.load("word2vec.model")

vector = model.wv['example']

# Find similar words
similar_words = model.wv.most_similar('example', topn=5)

print(similar_words)