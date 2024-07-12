import pickle
import requests

from bs4 import BeautifulSoup

import pandas as pd

from src.exception import CustomException
from src.logger import logger
from sklearn.metrics.pairwise import cosine_similarity


import pandas as pd



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
        storyline = a['storyline'] if not pd.isna(a['storyline']) else a['description']
        onlyTenItems[j] = {"title":title,"poster_url":poster_url,"genres":genres,"storyline":storyline}
        j += 1

    return onlyTenItems


