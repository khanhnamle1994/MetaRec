import joblib
import pandas as pd


class TitleRec:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.movies = joblib.load(path_to_artifacts + "movies.joblib")
        self.cosine_sim_title = joblib.load(path_to_artifacts + "cosine_sim_title.joblib")

    def pre_processing(self):
        # Get movie JSON data
        input_data = self.movies

        # JSON to Pandas dataframe
        input_data = pd.DataFrame(input_data)

        # Process 'title' column
        input_data['title'] = input_data['title'].fillna("").astype('str')

        return input_data

    def get_cosine_sim_title(self):
        # return the cosine similarity matrix stored in joblib object
        return self.cosine_sim_title

    def recommend(self, input_data, movie_title, cosine_sim_title):
        # Get all movie titles
        titles = input_data['title']
        # Get all movie indices
        indices = pd.Series(input_data.index, index=input_data['title'])

        # Get the index of the movie given as parameter
        idx = indices[movie_title["title"]]
        # Get similarity scores for the movie given as parameter
        sim_scores = list(enumerate(cosine_sim_title[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Return the top 10 movies with most similar scores
        sim_scores = sim_scores[1:11]

        # Return indices and titles of those 10 movies
        movie_indices = [i[0] for i in sim_scores]

        # Recommended titles
        recommended_titles = titles.iloc[movie_indices]

        return {"titles": recommended_titles, "status": "OK"}

    def get_recommendation(self, movie_title):
        try:
            input_data = self.pre_processing()
            cosine_sim_title = self.get_cosine_sim_title()
            recommendations = self.recommend(input_data, movie_title, cosine_sim_title)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return recommendations
