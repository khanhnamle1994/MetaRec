# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import re
import shutil
import warnings
import pandas as pd
from zipfile import ZipFile
from download_utils import maybe_download, download_path

# Default column names
DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_TIMESTAMP_COL = "timestamp"


class _DataFormat:
    def __init__(
        self,
        sep,
        path,
        has_header=False,
        item_sep=None,
        item_path=None,
        item_has_header=False,
    ):
        """MovieLens data format container as a different size of MovieLens data file has a different format
        Args:
            sep (str): Rating data delimiter
            path (str): Rating data path within the original zip file
            has_header (bool): Whether the rating data contains a header line or not
            item_sep (str): Item data delimiter
            item_path (str): Item data path within the original zip file
            item_has_header (bool): Whether the item data contains a header line or not
        """

        # Rating file
        self._sep = sep
        self._path = path
        self._has_header = has_header

        # Item file
        self._item_sep = item_sep
        self._item_path = item_path
        self._item_has_header = item_has_header

    @property
    def separator(self):
        return self._sep

    @property
    def path(self):
        return self._path

    @property
    def has_header(self):
        return self._has_header

    @property
    def item_separator(self):
        return self._item_sep

    @property
    def item_path(self):
        return self._item_path

    @property
    def item_has_header(self):
        return self._item_has_header


# Dictionary to store the data format
DATA_FORMAT = {
    "1m": _DataFormat(
        "::", "ml-1m/ratings.dat", False, "::", "ml-1m/movies.dat", False
    )
}

DEFAULT_HEADER = (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

# Warning and error messages
WARNING_MOVIE_LENS_HEADER = """MovieLens rating dataset has four columns (user id, movie id, rating, and timestamp),
but more than four column names are provided. Will only use the first four column names."""
ERROR_MOVIE_LENS_SIZE = "Invalid data size. Should be 1m."
ERROR_HEADER = "Header error. At least user and movie column names should be provided"


def load_pandas_df(
        size="1m",
        header=None,
        local_cache_path=None,
        title_col=None,
        genres_col=None,
        year_col=None,
):
    """Loads the MovieLens dataset as pd.DataFrame.
    Download the dataset from http://files.grouplens.org/datasets/movielens, unzip, and load.
    To load movie information only, you can use load_item_df function.
    Args:
        size (str): Size of the data to load.
        header (list or tuple or None): Rating dataset header.
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.
    Returns:
        pd.DataFrame: Movie rating dataset.

    **Examples**
    .. code-block:: python

        # To load just user-id, item-id, and ratings from MovieLens-1M dataset,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating'))
        # To load rating's timestamp together,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'))
        # To load movie's title, genres, and released year info along with the ratings data,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'),
            title_col='Title',
            genres_col='Genres',
            year_col='Year'
        )
    """
    size = size.lower()
    if size not in DATA_FORMAT:
        raise ValueError(ERROR_MOVIE_LENS_SIZE)

    if header is None:
        header = DEFAULT_HEADER
    elif len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]

    movie_col = header[1]

    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size))
        datapath, item_datapath = _maybe_download_and_extract(size, filepath)

        # Load movie features such as title, genres, and release year
        item_df = _load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )

        # Load rating data
        df = pd.read_csv(
            datapath,
            sep=DATA_FORMAT[size].separator,
            engine="python",
            names=header,
            usecols=[*range(len(header))],
            header=0 if DATA_FORMAT[size].has_header else None,
        )

        # Convert 'rating' type to float
        if len(header) > 2:
            df[header[2]] = df[header[2]].astype(float)

        # Merge rating df w/ item_df
        if item_df is not None:
            df = df.merge(item_df, on=header[1])

    return df


def _load_item_df(size, item_datapath, movie_col, title_col, genres_col, year_col):
    """Loads Movie info"""
    if title_col is None and genres_col is None and year_col is None:
        return None

    item_header = [movie_col]
    usecols = [0]

    # Year is parsed from title
    if title_col is not None or year_col is not None:
        item_header.append("title_year")
        usecols.append(1)

    if genres_col is not None:
        item_header.append(genres_col)
        usecols.append(2)  # genres column

    item_df = pd.read_csv(
        item_datapath,
        sep=DATA_FORMAT[size].item_separator,
        engine="python",
        names=item_header,
        usecols=usecols,
        header=0 if DATA_FORMAT[size].item_has_header else None,
        encoding="ISO-8859-1",
    )

    # Parse year from movie title. Note, MovieLens title format is "title (year)"
    # Note, there are very few records that are missing the year info.
    if year_col is not None:

        def parse_year(t):
            parsed = re.split("[()]", t)
            if len(parsed) > 2 and parsed[-2].isdecimal():
                return parsed[-2]
            else:
                return None

        item_df[year_col] = item_df["title_year"].map(parse_year)
        if title_col is None:
            item_df.drop("title_year", axis=1, inplace=True)

    if title_col is not None:
        item_df.rename(columns={"title_year": title_col}, inplace=True)

    return item_df


def _maybe_download_and_extract(size, dest_path):
    """Downloads and extracts MovieLens rating and item datafiles if they don’t already exist"""
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    _, rating_filename = os.path.split(DATA_FORMAT[size].path)
    rating_path = os.path.join(dirs, rating_filename)
    _, item_filename = os.path.split(DATA_FORMAT[size].item_path)
    item_path = os.path.join(dirs, item_filename)

    if not os.path.exists(rating_path) or not os.path.exists(item_path):
        download_movielens(size, dest_path)
        extract_movielens(size, rating_path, item_path, dest_path)

    return rating_path, item_path


def download_movielens(size, dest_path):
    """Downloads MovieLens datafile.
    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        dest_path (str): File path for the downloaded file
    """
    if size not in DATA_FORMAT:
        raise ValueError(ERROR_MOVIE_LENS_SIZE)

    url = "http://files.grouplens.org/datasets/movielens/ml-" + size + ".zip"
    dirs, file = os.path.split(dest_path)
    maybe_download(url, file, work_directory=dirs)


def extract_movielens(size, rating_path, item_path, zip_path):
    """Extract MovieLens rating and item datafiles from the MovieLens raw zip file.
    To extract all files instead of just rating and item datafiles,
    use ZipFile's extractall(path) instead.
    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        rating_path (str): Destination path for rating datafile
        item_path (str): Destination path for item datafile
        zip_path (str): zipfile path
    """
    with ZipFile(zip_path, "r") as z:
        with z.open(DATA_FORMAT[size].path) as zf, open(rating_path, "wb") as f:
            shutil.copyfileobj(zf, f)
        with z.open(DATA_FORMAT[size].item_path) as zf, open(item_path, "wb") as f:
            shutil.copyfileobj(zf, f)
