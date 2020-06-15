# Deploy Recommendation Models with Django

The source code for this Django web service is adapted from the tutorial available at [deploymachinelearning.com](https://deploymachinelearning.com)

This web service makes recommendation models available with REST API. Here are important properties:

- There can be several recommendation models available at the same endpoint with different versions. Furthermore, there can be many endpoint addresses defined.
- Information about requests sent to the models are stored. This can be used later for model testing and audit.
- There tests for research code and server code.
- You can run A/B tests between different versions of recommendation models.

## Code Structure

In the `research` directory there are:

- Code for [training content-based recommendation models](https://github.com/khanhnamle1994/transfer-rec/blob/master/Django-Web-Service/research/movielens_content_recommender.ipynb) on the MovieLens1M dataset.
- Code for [simulating A/B tests](https://github.com/khanhnamle1994/transfer-rec/blob/master/Django-Web-Service/research/ab_test.ipynb).

In the `backend` directory there is Django application.

In the `docker` directory, there are dockerfiles for running the service in the container.
