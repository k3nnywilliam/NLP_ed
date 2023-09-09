from gensim.models import Word2Vec

# Example corpus (replace with your own)
corpus = [
    ["this", "is", "an", "example", "sentence"],
    ["another", "example", "sentence"],
    ["yet", "another", "sentence"],
    # Add more sentences from your corpus
]

# Initialize and train the Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, sg=0)

# Save the trained model (optional)
model.save("word2vec.model")
