import pickle

def load_pickled_data2(
    title_df_path="title_df.pkl",
    cosine_similarity_paths=[
        "cosine_similarity1.pkl",
        "cosine_similarity2.pkl",
        "cosine_similarity3.pkl",
        "cosine_similarity4.pkl",
        "cosine_similarity5.pkl",
        "cosine_similarity6.pkl",
        "cosine_similarity7.pkl",
    ],
    df3_path="df3.pkl",
    filtered_genres_path="filtered_genres.pkl"
):

    data = {}

    # Load DataFrame
    with open(title_df_path, 'rb') as f:
        data["title_df"] = pickle.load(f)

    with open(df3_path, 'rb') as f:
        data["df3"] = pickle.load(f)

        # Load filtered genres
    with open(filtered_genres_path, 'rb') as f:
        data["filtered_genres"] = pickle.load(f)

    # Store cosine similarity paths
    data["cosine_similarity_paths"] = cosine_similarity_paths

    return data
