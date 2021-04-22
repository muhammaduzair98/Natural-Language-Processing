import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#Dataset
paragraph = """Hello Kese ho bro"""

sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer. lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
    print(sentences[i])

