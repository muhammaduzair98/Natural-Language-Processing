#Dataset
import pandas as pd

#corpus = pd.read_csv('dataset-CalheirosMoroRita-2017.csv')

corpus = open('dataset-CalheirosMoroRita-2017.csv', encoding = 'latin-1').read()
corpus = corpus.replace ("'", " ")


corpus = corpus.rstrip()
corpus = corpus.lstrip()
docs = corpus.split('\n')
docs

from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer( ngram_range=(1,1),min_df=20,max_df=200, max_features=100)

X = vec.fit_transform(docs)
print(X.shape)


X.toarray()


vec.get_feature_names()

vecFreq = vec.vocabulary_
vecFreq