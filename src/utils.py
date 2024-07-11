import pickle
import re
import bs4

import ast
import requests

from bs4 import BeautifulSoup

import pandas as pd

from src.exception import CustomException
from src.logger import logger
from sklearn.metrics.pairwise import cosine_similarity

from annoy import AnnoyIndex
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def calculate_cosine_similarity1(pca_data, X_pca, X_transformed):
    X_transformed.drop(["top_cast"],axis = 1, inplace = True)
    X_transformed['videos'] = X_transformed['videos'].fillna('[]')
    X_transformed['videos'] = X_transformed['videos'].apply(ast.literal_eval)
    X_transformed['poster_url'] = X_transformed['poster_url'].fillna('')
    annoy_index = AnnoyIndex(pca_data.shape[1], 'angular')  # 'angular' for cosine similarity approximation

    # Add items to Annoy index
    for i, vec in enumerate(pca_data):
        annoy_index.add_item(i, vec)

    annoy_index.build(10)  # 10 trees for better precision, adjust as needed

    similar_items = []
    onlyTenItems = {}

    # Query vector is X_pca, and we find nearest neighbors in pca_data
    for i, query_vector in enumerate(X_pca):
        neighbors = annoy_index.get_nns_by_vector(query_vector, 11)  # Retrieve top 11 (including itself)
        
        # Compute cosine similarity and store results
        for idx in neighbors[1:]:  # Skip itself
            sim = cosine_similarity([query_vector], [pca_data[idx]])[0, 0]
            similar_items.append((idx, sim))

    # Sort similar_items by similarity score in descending order
    similar_items.sort(key=lambda x: x[1], reverse=True)

    # Retrieve top 10 similar items
    top_10_similar_items = similar_items[:10]

    j = 0
    for idx, sim in top_10_similar_items:
        a = X_transformed.iloc[idx]
        title = a['title']
        poster_url = scrape_image_url(a['poster_url'])
        genres = a['genres'] if a['genres'] else ""
        videos = a['videos'][0]['link'] if a['videos'] else ""
        storyline = a['storyline'] if not pd.isna(a['storyline']) else a['description']

        onlyTenItems[j] = {
            "title": title,
            "poster_url": poster_url,
            "videos": videos,
            "genres": genres,
            "storyline": storyline
        }
        j += 1

    return onlyTenItems


def scrape_image_url(imdb_url):
    try:
        # Define headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Send a GET request to the IMDb URL with headers
        response = requests.get(imdb_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the meta tag with property="og:image"
            meta_tag = soup.find('meta', property='og:image')

            # Get the content attribute (image URL)
            if meta_tag:
                img_url = meta_tag['content']
                return img_url
            else:
                return ""
        else:
            return ""
    except Exception as e:
        print(f"Error fetching image URL: {e}")
        return ""

def concatenate_strings(row):
    str1 = row['title']
    str2_list = row['description']
    if str2_list == "[]": 
        return str1
    else:
        return str1 + " " + str2_list[3:]
    


def save_object(file_path,obj):
    try:
        with open(file_path,"wb") as file:
            pickle.dump(obj=obj,file=file)
    except Exception as e:
        raise CustomException("File not loaded properly")
    

def calculate_cosine_similarity(pca_data,X_pca,X_transformed):
    similar_items = []
    onlyTenItems = dict()
    X_transformed.drop(["top_cast"],axis = 1, inplace = True)
    X_transformed['videos'] = X_transformed['videos'].fillna('[]')
    X_transformed['videos'] = X_transformed['videos'].apply(ast.literal_eval)
    X_transformed['poster_url'] = X_transformed['poster_url'].fillna('')
    for i in range(len(pca_data)):
        sim = cosine_similarity(pca_data[i].reshape(1, -1), X_pca.reshape(1, -1))[0, 0]
        similar_items.append((i, sim))

    # Sort similar_items by similarity score in descending order
    similar_items.sort(key=lambda x: x[1], reverse=True)

    # Retrieve top 10 similar items
    top_10_similar_items = similar_items[:10]
    j = 0
    for idx, sim in top_10_similar_items:
                #if(final_df['store'].iloc[idx] == final_df['store'].iloc[item_index]):
        a = X_transformed.iloc[idx]
        logger.info(f"{a}")
        title = a['title']
        poster_url = scrape_image_url(a['poster_url'])
        if type(a['genres']) == float:
            genres = ""
        else:
            print(a['genres'])
            genres = a['genres']
        if a['videos'] == []:
            videos = ""
        else:
            videos = a['videos'][0]['link']
        storyline = a['storyline'] if not pd.isna(a['storyline']) else a['description']
        onlyTenItems[j] = {"title":title,"poster_url":poster_url,"videos":videos,"genres":genres,"storyline":storyline}
        j += 1

    return onlyTenItems


def shortDesc(x):
  if len(x) > 1000:
    return x[:1000]
  else:
    return x