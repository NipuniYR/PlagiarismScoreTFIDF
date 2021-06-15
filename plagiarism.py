# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 06:14:19 2021

@author: Nipuni
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

all_files = os.listdir("data/")

#corpus
documents = []

for f in all_files:
    with open("./data/"+f,encoding='utf-8') as file:
        line = file.read().replace("\n"," ")
        documents.append(line)
        
#tf-idf
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)
featureNames = vectorizer.get_feature_names()

print("################Output of the vectorizer.fir_transform(documents)################")
print(tfidf)
print()
denseTfidf = tfidf.todense()
w = denseTfidf.nonzero()
denseTfidfLabeled = pd.DataFrame(denseTfidf,index=all_files,columns=featureNames)

#Cosine similarity
cosine_similarities = cosine_similarity(tfidf[5,:], tfidf)
cosine_similaritiesLabeled = pd.DataFrame(cosine_similarities,columns=all_files)

#Plagiarism scores as a percentage
scores = cosine_similaritiesLabeled*100




##################################### TF-IDF #####################################
# TF - Term Frequency 
# TF[word1 in doc1] = (Total number of occurances of word1 in the doc1
#                                               /Total number of words in doc1)
# 
# IDF - Inverse Document Frequency
# IDF[word1] = log(Total number of documents/Number of documents which contain word1)
#
# TF-IDF
# TF-IDF[word1 in doc1] = TF[word1 in doc1]*IDF[word1]

############################### Cosine Similarity ###############################
# Cosine of the angle between two vectors (has a value between 0 and 1) 
#                              - measure similarity between two vectors
# 
# Cos(0 degree) = 1 (when the angle between the two venctors is 0, 
#                          the two vectors are more similar to each other)