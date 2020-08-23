# Import libraries
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F

# Import utility scripts
from indexes import create_user_index, create_doc_index, load_indexes, map_recommendations
from configs import data_path


def read_data(sc, data_file, delimiter='::'):
    """
    Read the data into an RDD of tuple (usrId, productId, rating).
    :param sc: An active SparkContext.
    :param data_file: A (delimiter) separated file.
    :param delimiter: The delimiter used to separate the 3 fields of the input file. Default is ','.
    :return: ui_mat_rdd: The UI matrix in an RDD.
    """
    data = sc.textFile(data_file)
    header = data.first()

    def _func(x):
        return int(x.split(delimiter)[0]), int(x.split(delimiter)[1]), float(x.split(delimiter)[2])

    ui_mat_rdd = data.filter(lambda row: row != header).map(_func)
    return ui_mat_rdd


if __name__ == "__main__":
    # Initialize Spark sessions
    sc = SparkContext()
    spark = SparkSession(sc)
    # Read in the data
    ui_mat_rdd = read_data(sc, data_path, delimiter='::').sample(False, 1.0, seed=0).persist()

    print('Creating usr and doc indexes...')
    user_index = create_user_index(ui_mat_rdd)
    doc_index = create_doc_index(ui_mat_rdd)
    b_uidx = sc.broadcast(user_index)
    b_didx = sc.broadcast(doc_index)

    def _func(i):
        usrId, docId, value = i
        return b_uidx.value[usrId], b_didx.value[docId], value

    ui_mat_rdd = ui_mat_rdd.map(_func)

    def _func(i):
        usrId, docId, value = i
        return usrId

    num_users = ui_mat_rdd.map(_func).distinct().count()

    def _func(i):
        usrId, docId, value = i
        return docId

    num_movies = ui_mat_rdd.map(_func).distinct().count()
    print('users:', num_users, 'products:', num_movies)

    # Create Spark dataframe
    df = spark.createDataFrame(ui_mat_rdd, ['userId', 'movieId', 'value'])

    ui_mat_rdd.unpersist()

    print('Splitting data set...')
    df = df.orderBy(F.rand())

    train_df, test_df = df.randomSplit([0.9, 0.1], seed=45)
    train_df, val_df = train_df.randomSplit([0.95, 0.05], seed=45)

    train_df = train_df.withColumn('flag', F.lit(0))
    val_df = val_df.withColumn('flag', F.lit(1))
    val_df = val_df.union(train_df)
    test_df = test_df.withColumn('flag', F.lit(2))
    test_df = test_df.union(train_df)
    test_df = test_df.union(val_df)

    train_size = train_df.count()
    val_size = val_df.count()
    test_size = test_df.count()

    train_df.show()
    print(train_size, 'training examples')
    print(val_size, 'validation examples')
    print(test_size, 'testing example')

    train_examples = train_df.select(
        "movieId", F.struct(["userId", "value",
                             "flag"]).alias("ranking")).groupby('movieId').agg(
                                 F.collect_list('ranking').alias('rankings'))
    val_examples = val_df.select(
        "movieId", F.struct(["userId", "value",
                             "flag"]).alias("ranking")).groupby('movieId').agg(
                                 F.collect_list('ranking').alias('rankings'))
    test_examples = test_df.select(
        "movieId", F.struct(["userId", "value",
                             "flag"]).alias("ranking")).groupby('movieId').agg(
                                 F.collect_list('ranking').alias('rankings'))

    train_examples.show()
    val_examples.show()
    test_examples.show()

    train_examples.coalesce(1).write.json(
        path="data/train_set", mode='overwrite')
    val_examples.coalesce(1).write.json(path="data/val_set", mode='overwrite')
    test_examples.coalesce(1).write.json(
        path="data/test_set", mode='overwrite')
