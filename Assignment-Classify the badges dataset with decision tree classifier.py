#Python Based Text Mining | Udemy Course | By: Dr. Taimoor
#Assignment 2
#Training multiple models on a same dataset and algorithm.


'''Following are the steps to be performed.
1) download dataset
2) Read into a variable and separate on  new line
3) Structure the dataset using any of the vectorizer
4) import decision tree classifier and create an object of it
5) Train the model on all instances except the last 10
6) Evaluate the model on the last 10 instances (possible values for maximum tree depth are 2, 3 & 4)'''

corpus = open('Desktop/NLP/badges.data.txt').read()
#corpus

docs = corpus.split('\n')
docs

X = [] #Labels
y = [] #Names

for doc in docs:
    l = doc[:1]
    i = doc[2:]
    X.append(i)
    y.append(l)


from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
matrix_X = tf.fit_transform(X)

matrix_X

### Training DecisionTree Model

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=10)

dt.fit(matrix_X[:284], y[:284])

dt.predict(matrix_X[284:])

dt.predict_proba(matrix_X[284:])

dt.score(matrix_X, y)

### Training Naive Bayes Model For Same Dataset

from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()

NB.fit(matrix_X[:284], y[:284])

NB.predict(matrix_X[284:])

NB.predict_proba(matrix_X[284:])

### Training KNN For Same Dataset

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=9)

KNN.fit(matrix_X[:284],y[:284])

KNN.predict(matrix_X[284:])

KNN.predict_proba(matrix_X[284:])

### Training Linear Regression Model For Same Dataset

from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier(alpha=0.01, max_iter=20)

SGD.fit(matrix_X[:284], y[:284])

SGD.predict(matrix_X[284:])

### SGD Model Doesn't worked correctly, even after lots of tweaks too.