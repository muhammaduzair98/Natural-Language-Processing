import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#dataset

paragraph = """In 2001, Apple released Mac OS X, a modern Unix-based operating system which was later rebranded to simply OS X in 2012, and then macOS in 2016. The current version is macOS Catalina, released on October 7, 2019.[6] Intel-based Macs are capable of running native third party operating systems such as Linux, FreeBSD, and Microsoft Windows with the aid of Boot Camp or third-party software. Volunteer communities have customized Intel-based macOS to run illicitly on non-Apple computers."""


#Objects

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

sentences = nltk.sent_tokenize(paragraph)

corpus = []

for i in range (len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) 
              for word in review 
              if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

corpus

#Creating Bag of Words Model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

X

