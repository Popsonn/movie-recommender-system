import streamlit as st

from recommendation import generate_movie_recommendations
from fetch_movie_info import fetch_movie_info
from load_data2 import load_pickled_data2

# Load data and get cosine similarity paths
@st.cache
def load_data2():
    return load_pickled_data2()

@st.cache
def cached_fetch_movie_info(tmdbId):
    return fetch_movie_info(tmdbId)

@st.cache
def cached_generate_movie_recommendations(movie_title, full_movie_titles_df, cosine_similarity_paths, num_recommendations=10):
    return generate_movie_recommendations(movie_title, full_movie_titles_df, cosine_similarity_paths, num_recommendations)


# Load data and get cosine similarity paths
data = load_pickled_data2()
title_df = data["title_df"]
df3 = data["df3"]
cosine_similarity_paths = data["cosine_similarity_paths"]

def main():
    st.set_page_config(page_title="Recommender System", layout="wide")
    st.title('Movie Recommender System')
    st.image('C:/Users/DELL/Downloads/1_oRJYoC18Msc4tC6FExYRKQ.jpg')

    # Add an introductory paragraph with additional information and feedback request
    st.markdown("""
# Welcome to My Movie Recommender System!

This app helps you discover hidden gems and revisit old favorites! Explore a vast movie library spanning decades of cinema, including recent releases(**up to 2019**).

Uncover hidden gems and rediscover old favorites based on your tastes! We use your past preferences to recommend similar movies you might enjoy.

**Let us know what you think!** We're always looking for ways to improve the recommendations.
""")

    unique_movie_titles = title_df['title'].unique()

    selected_movie = st.selectbox(
        '**Which Movie Do you like?**',
        unique_movie_titles
    )

    num_recommendations = st.slider('**Number of recommendations:**', 1, 20, 10)

    if st.button("**Get Recommendations**"):
        with st.spinner('Generating recommendations...'):
            # Fetch movie information using the TMDB ID
            try:
                selected_movie_id = title_df.loc[title_df['title'] == selected_movie, 'movieId'].values[0]
                # Use the movie ID to look up the TMDb ID in the links dataset
                tmdbId = df3[df3['movieId'] == selected_movie_id]['tmdbId'].iloc[0]
                movie_info = fetch_movie_info(tmdbId)
            except (KeyError, IndexError):
                st.write("Sorry, the movie information for the selected movie is missing.")
                movie_info = None

            # Display movie information (unchanged)
            if movie_info:
                st.write("### Movie Information")
                st.write(f"**Title:** {movie_info['title']}")
                st.write(f"**Rating:** {movie_info['rating']}")
                st.write(f"**Genres:** {', '.join(movie_info['genres'])}")
                st.write(f"**Overview:** {movie_info['overview']}")
                st.image(movie_info['poster_url'])
            else:
                st.write("Sorry, no movie information found for the selected movie.")

            # Generate and display recommendations
            try:
                recommendations = generate_movie_recommendations(
                    selected_movie, title_df['title'], cosine_similarity_paths, num_recommendations=num_recommendations
                )
            except (KeyError, IndexError) as e:
                # Provide more specific error messages based on the exception type
                if isinstance(e, KeyError):
                    st.write(f"An error occurred: Movie '{selected_movie}' might be missing from our data.")
                elif isinstance(e, IndexError):
                    st.write(f"An error occurred while processing recommendations. Please try again later.")
                else:
                    st.write(f"An unexpected error occurred: {e}")
                recommendations = None

            if recommendations:
                st.subheader('Similar Movies')

                titles = [rec[0] for rec in recommendations]
                if len(titles) > 3: 
                    first_three_titles = titles[:3]
                    st.markdown(f"If you like **{selected_movie}**, you might also like: **{', '.join(first_three_titles[:-1])}** and **{first_three_titles[-1]}**")
                else:
                    st.markdown(f"If you like **{selected_movie}**, you might also like: {', '.join(titles)}")
                num_columns = min(num_recommendations, 5) # Maximum of 5 columns
                num_rows = (num_recommendations + num_columns - 1) // num_columns 
                for row_index in range(num_rows):
                    cols = st.columns(num_columns)
                    for col_index in range(num_columns):
                        recommendation_index = row_index * num_columns + col_index
                        if recommendation_index < num_recommendations:
                            with cols[col_index]:

                                st.markdown("""<style>
                                            .movie-info {
                                                max-height: 254px;
                                                width: 148.22px;
                                                overflow-y: auto;
                                            }
                                        </style>""", unsafe_allow_html=True)
                                movie_title = recommendations[recommendation_index][0]
                                movie_id = title_df[title_df['title'] == movie_title]['movieId'].iloc[0]
                                tmdb_id = df3[df3['movieId'] == movie_id]['tmdbId'].iloc[0]
                    # Fetch movie information using TMDB API
                                movie_info = fetch_movie_info(tmdb_id)
                                if movie_info:
                                    st.image(movie_info['poster_url'], width=148.22)
                                    st.markdown(f"<div class='movie-info'>"
                            f"<p><strong>Title:</strong> {movie_info['title']}</p>"
                            f"<p><strong>Rating:</strong> {movie_info['rating']}</p>"
                            f"<p><strong>Genres:</strong> {', '.join(movie_info['genres'])}</p>"
                            f"<p><strong>Overview:</strong> {movie_info['overview']}</p>"
                            "</div>", unsafe_allow_html=True)
                                else:
                                    st.write("No tmdb information found")
            else:
                st.write(f"Sorry, '{selected_movie}' was not found in our database.")

if __name__ == "__main__":
    main()

#st.sidebar.markdown("### Connect on:")
#st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/deborahpopoola/)")
#st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Popsonn)")
