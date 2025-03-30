import pandas as pd
import numpy as np
import openai
from dotenv import dotenv_values
import json
from pinecone import Pinecone, ServerlessSpec



###


# Load API key
config = dotenv_values(".env")
api_key = config["OPENAI_API_KEY"]
openai.api_key = api_key

import numpy as np


def embed(docs: list[str]) -> list[list[float]]:
    res = openai.embeddings.create(
        input=docs,
        model="text-embedding-3-small"
    )
    doc_embeds = [r.embedding for r in res.data]
    return doc_embeds


if __name__ == "__main__":

    # query_movie_name = 'lord of the rings'
    # query_movie_name = 'Harry pottser'
    query_movie_name = 'bridgitte jones'
    query_movie_name_embedded = embed(query_movie_name)[0]

    ## initialize pinecone
    pc = Pinecone(api_key="pcsk_2mQKt1_DpveutRZj7oRErXmMYdTY8hhoCH36Z4TyFaGBkyTZ3gVMk6GB8TrDg2qA7wwEsD")

    ## find movies with similar names and get the id
    movie_name_index_name = "movies-names-index"
    index_movie_names = pc.Index(movie_name_index_name)
    response = index_movie_names.query(vector=query_movie_name_embedded, top_k=2, include_metadata=True)
    movie_id = response.matches[0]['id']

    ## use the movies description index to find similar movies
    index_name = "movies-index"
    index = pc.Index(index_name)
    fetched_vector = index.fetch(ids=[movie_id])
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

        # Step 3: Print results
        for match in search_results["matches"]:
            print(f"Movie: {match['metadata']['name']}, Rating: {match['metadata']['rating']}")
            print(f"Similarity Score: {match['score']}")
            print(f"Movie Description': {match['metadata']['description']}")
            print("*"*20)

    else:
        print("Movie ID not found in Pinecone")
