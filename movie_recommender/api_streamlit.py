import streamlit as st
import pinecone

from movie_recommendation_module import MovieRecommender


# Streamlit UI
st.title("🎬 Movie Recommender App")

# User input
movie_name = st.text_input("Enter a movie name:")
st.write(movie_name)

movie_recommender = MovieRecommender()

if st.button("Find Similar Movies") and movie_name:
    print('1')
    with st.spinner("Searching for recommendations..."):
        # Return specific movie by ID
        print('2')
        print(movie_name)
        movie_id = movie_recommender.find_movie_in_db(movie_name)
        st.write(movie_id)
        recommendations, full_score = movie_recommender.find_similar_movies(movie_id)

        # Step 3: Print results
        if recommendations:
            for i, match in enumerate(recommendations["matches"]):
                st.write(f"Movie: {match['metadata']['name']}, Rating: {match['metadata']['rating']}")
                st.write(f"Similarity Score: {match['score']}")
                st.write(f"Similarity+Rating Score: {full_score[i]}")

                st.write(f"Movie Description': {match['metadata']['description']}")
                st.write("*" * 20)
        else:
            st.warning("No similar movies found. Try another title!")

        # if recommendations:
        #     st.subheader("Recommended Movies:")
        #     for movie in recommendations:
        #         st.write(f"🎥 **{movie['metadata']['name']}** (Similarity: {movie['score']:.2f})")
        # else:
        #     st.warning("No similar movies found. Try another title!")
