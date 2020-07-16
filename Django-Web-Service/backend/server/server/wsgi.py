"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.movie_rec.content_rec import ContentRec
from apps.ml.movie_rec.title_rec import TitleRec

try:
    # create ML registry
    registry = MLRegistry()
    # content based recommendation
    cr = ContentRec()
    # add to ML registry
    registry.add_algorithm(endpoint_name="movie_rec",
                           algorithm_object=cr,
                           algorithm_name="content-based recommendation",
                           algorithm_status="production",
                           algorithm_version="0.0.2",
                           owner="James",
                           algorithm_description="Content-Based Recommendation based on Movie Genres",
                           algorithm_code=inspect.getsource(ContentRec))

    # title based recommendation
    tr = TitleRec()
    # add to ML registry
    registry.add_algorithm(endpoint_name="movie_rec",
                           algorithm_object=tr,
                           algorithm_name="title-based recommendation",
                           algorithm_status="testing",
                           algorithm_version="0.0.2",
                           owner="James",
                           algorithm_description="Content-Based Recommendation based on Movie Titles",
                           algorithm_code=inspect.getsource(TitleRec))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
