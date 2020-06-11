from django.test import TestCase
from rest_framework.test import APIClient


class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
        movie_title = {
            "title": "Good Will Hunting (1997)"
        }
        classifier_url = "/api/v1/movie_rec/recommend?status=production&version=0.0.2"
        response = client.post(classifier_url, movie_title, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["titles"]), 10)
        self.assertTrue("titles" in response.data)
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)
