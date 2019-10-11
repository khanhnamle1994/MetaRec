import tensorflow as tf
import os

def _get_training_data(FLAGS):
    ''' Building the input pipeline for training and inference using TFRecords files.
    @return data only for the training
    @return data for the inference
    '''

    filenames=[FLAGS.tf_records_train_path+f for f in os.listdir(FLAGS.tf_records_train_path)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    dataset2 = tf.data.TFRecordDataset(filenames)
    dataset2 = dataset2.map(parse)
    dataset2 = dataset2.shuffle(buffer_size=1)
    dataset2 = dataset2.repeat()
    dataset2 = dataset2.batch(1)
    dataset2 = dataset2.prefetch(buffer_size=1)

    return dataset, dataset2

def _get_test_data(FLAGS):
    ''' Building the input pipeline for test data.'''

    filenames=[FLAGS.tf_records_test_path+f for f in os.listdir(FLAGS.tf_records_test_path)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset

def parse(serialized):
    ''' Parser fot the TFRecords file.'''

    features = {'movie_ratings':tf.FixedLenFeature([3952], tf.float32),}

    parsed_example = tf.parse_single_example(serialized, features=features,)

    movie_ratings = tf.cast(parsed_example['movie_ratings'], tf.float32)

    return movie_ratings
