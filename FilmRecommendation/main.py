import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

credits = pd.read_csv("dataset/tmdb_5000_credits.csv")
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits_columns_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_df_merge = movies.merge(credits_columns_renamed, on="id")
movies_cleaned_df = movies_df_merge.drop(
    columns=["homepage", "title_x", "title_y", "status", "production_countries"]
)
movies_cleaned_df["overview"] = movies_cleaned_df["overview"].fillna("")
movies = movies_cleaned_df

tfv = TfidfVectorizer(
    min_df=3,
    max_features=None,
    strip_accents="unicode",
    analyzer="word",
    token_pattern="\w{1}",
    ngram_range=(1, 3),
    stop_words="english",
)

tfv_matrix = tfv.fit_transform(movies["overview"])
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()

def give_rec(title, sig=sig):
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    movie_indices = [i[0] for i in sig_scores]
    return movies['original_title'].iloc[movie_indices]

give_rec('The Dark Knight Rises')

print(movies.head(10)['original_title'])