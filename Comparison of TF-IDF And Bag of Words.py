import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

#Dataset
paragraph = """Apple has developed a series of Macintosh operating systems. The first versions initially had no name but came to be known as the "Macintosh System Software" in 1988, "Mac OS" in 1997 with the release of Mac OS 7.6, and retrospectively called "Classic Mac OS". Apple produced a Unix-based operating system for the Macintosh called A/UX from 1988 to 1995, which closely resembled contemporary versions of the Macintosh system software. Apple does not license macOS for use on non-Apple computers, however, System 7 was licensed to various companies through Apple's Macintosh clone program from 1995 to 1997. Only one company, UMAX Technologies was legally licensed to ship clones running Mac OS 8.[5]

In 2001, Apple released Mac OS X, a modern Unix-based operating system which was later rebranded to simply OS X in 2012, and then macOS in 2016. The current version is macOS Catalina, released on October 7, 2019.[6] Intel-based Macs are capable of running native third party operating systems such as Linux, FreeBSD, and Microsoft Windows with the aid of Boot Camp or third-party software. Volunteer communities have customized Intel-based macOS to run illicitly on non-Apple computers.

The Macintosh family of computers have operated using a variety of different CPU architectures since its introduction. Originally they used the Motorola 68000 series of microprocessors. In the mid 1990s they transitioned to PowerPC processors, and again in the mid 2000s they began to use 32- and 64-bit Intel x86 processors. Apple has confirmed that it will be transitioning CPU architectures again, this time to its own ARM-based processors for use in the Macintosh beginning in 2020.[7]"""

#Declaring Objects & Corpus Array

sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()
corpus = []

#Cleaning The Text

for i in range(len(sentences)):
    review = re.sub('[a-zA-z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    

corpus

#Creating the TF-IDF Model

print ("Dataset TF-IDF Model")

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)

#Creating Bag of Words Model

print ("Dataset Bag of Words Model")

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)



