from django.test import TestCase
from apps.ml.movie_rec.content_rec import ContentRec

import inspect
from apps.ml.registry import MLRegistry


class MLTests(TestCase):
    def test_content_rec_algorithm(self):
        movie_title = {
            "title": "Good Will Hunting (1997)"
        }

        # Initialize model class
        my_alg = ContentRec()
        response = my_alg.get_recommendation(movie_title)
        self.assertEqual('OK', response['status'])
        self.assertTrue('titles' in response)

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "movie_rec"
        algorithm_object = ContentRec()
        algorithm_name = "content-based recommendation"
        algorithm_status = "production"
        algorithm_version = "0.0.2"
        algorithm_owner = "James"
        algorithm_description = "Content-Based Recommendation based on Movie Genres"
        algorithm_code = inspect.getsource(ContentRec)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                               algorithm_status, algorithm_version, algorithm_owner,
                               algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
