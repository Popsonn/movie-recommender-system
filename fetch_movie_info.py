import requests

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
