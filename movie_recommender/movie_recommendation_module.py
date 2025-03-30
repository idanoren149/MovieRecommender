import pandas as pd
import numpy as np
import openai
from dotenv import dotenv_values
import json
from pinecone import Pinecone, ServerlessSpec
import streamlit as st


# Load API key
config = dotenv_values(".env")
# api_key = config["OPENAI_API_KEY"]
# openai.api_key = api_key
openai.api_key = st.secrets["openai"]["api_key"]


pinecone_api_key = config['PINECONE_API_KEY']

class MovieRecommender():
    def __init__(self):
        ## initialize pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)

    # @staticmethod
    def embed(self, docs: list[str]) -> list[list[float]]:
        res = openai.embeddings.create(
            input=docs,
            model="text-embedding-3-small"
        )
        doc_embeds = [r.embedding for r in res.data]
        return doc_embeds

    def find_movie_in_db(self, query_movie_name):
        query_movie_name_embedded = self.embed(query_movie_name)[0]

        ## find movies with similar names and get the id
        movie_name_index_name = "movies-names-index"
        index_movie_names = self.pc.Index(movie_name_index_name)
        response = index_movie_names.query(vector=query_movie_name_embedded, top_k=2, include_metadata=True)
        movie_id = response.matches[0]['id']

        return movie_id

    def find_similar_movies(self, movie_id):
        index_name = "movies-index"
        index = self.pc.Index(index_name)
        fetched_vector = index.fetch(ids=[movie_id])
        # print(f'fetched_vector {fetched_vector}')

        # fetched_vector = fetched_vector.vectors[id].values

        # Check if the vector exists
        if fetched_vector and movie_id in fetched_vector.vectors:
            query_vector = fetched_vector.vectors[movie_id]["values"]

            # Step 2: Perform similarity search (top 5 similar movies)
            search_results = index.query(
                vector=query_vector,
                top_k=7,  # Get top 5 similar movies
                include_metadata=True  # Include movie details
            )

            # Sorting by similarity score + rating
            matches = search_results["matches"]
            full_score = []
            for match in matches:
                full_score.append(match['score']*10 + match['metadata']['rating'])

            index_sort = np.argsort(full_score)[::-1]
            full_score = np.array(full_score)[index_sort]

            # print(type(index_sort))
            # print(index_sort)
            # print(type(matches))
            # print(matches)
            # print(match)
            # matches = np.array(matches)[index_sort]

            # Step 3: Print results
            # for match in matches:
            for index in index_sort:
                match = matches[index]
                print(f"Movie: {match['metadata']['name']}, Rating: {match['metadata']['rating']}")
                print(f"Similarity Score: {match['score']}")
                print(f"Similarity+Rating Score: {full_score[index]}")
                print(f"Movie Description': {match['metadata']['description']}")
                print("*" * 20)

        else:
            print("Movie ID not found in Pinecone")

        matches_sorted = []
        for index in index_sort:
            matches_sorted.append(matches[index])

        search_results['matches'] = matches_sorted

        return search_results, full_score


