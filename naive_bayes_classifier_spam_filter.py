import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

import re
stop_words = stopwords.words('english')
stemmer = nltk.SnowballStemmer("english")

# Load your dataset
data = pd.read_csv('data/spam.csv')

# Split the dataset into training and testing sets
X = data['text']
y = data['label']

def clean_text(text):
    sms = re.sub('[^a-zA-Z]', ' ', text) #Replacing all non-alphabetic characters with a space
    sms = sms.lower() #converting to lowecase
    sms = sms.split()
    sms = ' '.join(sms)
    return sms


def preprocess(text):
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

data['text'] = data['text'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=100)
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)

# Predict labels for the test set
y_pred = nb_classifier.predict(X_test_counts)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Classifier Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Classify new messages
new_messages = ["Congratulations, you've won a free vacation!",
                "Can we schedule a meeting for next week?",
                "Hey baby, I would like to have some fun. Call me or go to www.rabbithump.com for fun times!",
                "You have won the jackpot! please enter your credit card!"]

new_messages_counts = vectorizer.transform(new_messages)
new_messages_pred = nb_classifier.predict(new_messages_counts)

print(f"\nNew Messages Classification:")
for message, label in zip(new_messages, new_messages_pred):
    print(f"Message: {message}\nClassification: {label}\n")

