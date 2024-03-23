import streamlit as st
import requests
import pickle

st.set_page_config(page_title="Recommender System", layout="wide")
st.title('Movie Recommender System')
st.image('1_oRJYoC18Msc4tC6FExYRKQ.jpg')

    # Add an introductory paragraph with additional information and feedback request
st.markdown("""
# Welcome to My Movie Recommender System!

This app helps you discover hidden gems and revisit old favorites! Explore a vast movie library spanning decades of cinema, including recent releases(**up to 2019**).

Uncover hidden gems and rediscover old favorites based on your tastes! We use your past preferences to recommend similar movies you might enjoy.

**Let us know what you think!** We're always looking for ways to improve the recommendations.
""")

@st.cache_data
def fetch_movie_info(tmdbId):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{tmdbId}?api_key=f68f3ccf3f4513470631fda0eed818ab&language=en-US")
    if response.status_code == 200:
        data = response.json()
        genres = [genre['name'] for genre in data.get('genres', [])]  
        movie_info = {
            'title': data.get('title'),
            'overview': data.get('overview'),
            'poster_url': f"https://image.tmdb.org/t/p/w500/{data.get('poster_path')}",
            'rating': data.get('vote_average'),
            'genres': genres
            # Add more fields as needed
        }
        return movie_info
    else:
        return None

@st.cache_data
def generate_movie_recommendations(movie_title, full_movie_titles_df, cosine_similarity_paths, num_recommendations, similarity_threshold=0.6):
  """
  Generates movie recommendations for a given movie title using multiple cosine similarity matrices.

  Args:
      movie_title (str): Title of the target movie.
      full_movie_titles_df (pandas.DataFrame): DataFrame containing movie titles.
      cosine_similarity_paths (List[str]): List of paths to the pickled cosine similarity matrices.
      num_recommendations (int): Number of recommendations to generate.
      similarity_threshold (float): Minimum similarity score for recommendations.

  Returns:
      List[Tuple[str, float]]: List of recommended movies and their corresponding similarity scores.
  """

  # Try to find the target movie's index in the DataFrame
  try:
      matching_titles = full_movie_titles_df == movie_title
      target_movie_index = matching_titles.where(matching_titles).idxmax()

      # Initialize a dictionary to store aggregated similarity scores for each movie
      movie_scores = {}

      # Iterate over each cosine similarity path
      for cosine_similarity_path in cosine_similarity_paths:
          # Load cosine similarity matrix on demand
          with open(cosine_similarity_path, 'rb') as f:
              cosine_similarity = pickle.load(f)

          # Get the target movie's similarity scores with other movies
          movie_similarities = cosine_similarity[target_movie_index]

          # Iterate over movies and their similarity scores
          for index, similarity_score in enumerate(movie_similarities):
              # Exclude the target movie itself
              if index != target_movie_index:
                  # Update or add similarity score for the movie
                  movie_scores[index] = movie_scores.get(index, 0) + similarity_score

      # Sort movie scores by overall similarity (descending order)
      sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

      # Select top `num_recommendations` similar movies
      recommended_indices = [index for index, _ in sorted_movie_scores[:num_recommendations]]

      # Extract recommended movie titles and their aggregated similarity scores
      recommendations = [(full_movie_titles_df.iloc[index], movie_scores[index]) for index in recommended_indices]

      return recommendations

  except (KeyError, IndexError):
      print(f"Sorry, the movie '{movie_title}' you requested is not in our database. Please check the spelling or try with some other movie.")
      return []



@st.cache_data
def load_pickled_data2(
    title_df_path="C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/title_df.pkl",
    cosine_similarity_paths=[
        "C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/cosine_similarity1.pkl",
        "C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/cosine_similarity2.pkl",
        "C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/cosine_similarity3.pkl",
        "C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/cosine_similarity4.pkl",
        "C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/cosine_similarity5.pkl",
        "C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/cosine_similarity6.pkl",
        "C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/cosine_similarity7.pkl",
    ],
    df3_path="C:/Users/DELL/Downloads/ml-25m (1)/ml-25m/df3.pkl"
):

    data = {}

    # Load DataFrame
    with open(title_df_path, 'rb') as f:
        data["title_df"] = pickle.load(f)

    with open(df3_path, 'rb') as f:
        data["df3"] = pickle.load(f)

        # Load filtered genres

    # Store cosine similarity paths
    data["cosine_similarity_paths"] = cosine_similarity_paths

    return data

data = load_pickled_data2()
title_df = data["title_df"]
df3 = data["df3"]
cosine_similarity_paths = data["cosine_similarity_paths"]
unique_movie_titles = title_df['title'].unique()



def main():

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
