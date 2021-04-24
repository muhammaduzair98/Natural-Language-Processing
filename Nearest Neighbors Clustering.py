### In this assignment, students have to use the Eco-hotel dataset from the UCI repository and perform nearest neighbors clustering on it using radius based condition rather than n_neighbors.

###### Following are the instructions for the assignment,
1) Download the dataset from the UCI repository
2) Read the data in a variable and separate on new line
3) Vectorize the data using a vectorizer
4) Train the nearest neighbors with different radius values
5) Check the cluster labels to see number of clusters.

#Dataset

import pandas as pd

corpus = pd.read_csv('/Users/eapple/Desktop/NLP/dataset2.csv', encoding = "Latin-1", delimiter = '\n')
raw_data= corpus.values
#raw_data


#dataset = [raw_data[i][0]for i in range (0, len(raw_data))]


#newCorpus = open('/Users/eapple/Desktop/NLP/dataset2.csv', encoding='Latin-1').read()
#newCorpus.split('\n')

data = corpus.values
data

clean_data = [data[i][0] for i in range (0, len(data))]
clean_data

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
matrix_input = vec.fit_transform(clean_data)
matrix_input

from sklearn.neighbors import NearestNeighbors 

NN = NearestNeighbors()
NN

NN.fit(matrix_input)
NN.kneighbors(matrix_input[0], 5)

NN.radius_neighbors(matrix_input[0], 5)

NN = NearestNeighbors(radius=0.0001)
NN.fit(matrix_input)
NN.kneighbors(matrix_input[0],5)


