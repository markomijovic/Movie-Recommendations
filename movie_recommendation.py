import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# helper functions
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

likedMovie = "GoodFellas"
### read the movie database
df = pd.read_csv("movie_database.csv", engine ='python')
### select movie information to be used in the algorithm
features = ['keywords', 'cast', 'genres', 'director']
### delete NaNs
for feature in features:
    df[feature] = df[feature].fillna('')
### combine all selected features into 1 rows
def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
    except:
        print("Error: ", row)
df["combined_features"] = df.apply(combine_features, axis = 1)
### count matrix/cosine similarity matrix
cv = CountVectorizer()
countMatrix = cv.fit_transform(df["combined_features"])
cosSim = cosine_similarity(countMatrix)
try:
    movieIndex = int(get_index_from_title(likedMovie))
except:
    raise NameError("Movie name not found.")
# a sorted list of tuples of form (movie index, similarity score)
similarMovies = list(enumerate(cosSim[movieIndex]))
sortedSimilarMovies = sorted(similarMovies, key=lambda x:x[1], reverse=True)
for i in range(0, 50):
   print(get_title_from_index(sortedSimilarMovies[i][0]))


