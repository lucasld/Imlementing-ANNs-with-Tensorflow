import tensorflow as tf


class Data_Preparation:
    """Class that manages data generation, splitting and preprocessing.
    
    :param sequence_length: length of the sequence of inputs
    :type sequence_length: integer
    :param dataset_length: length of the dataset
    :type dataset_length: integer
    """
    def __init__(self, sequence_length, dataset_length) -> None:
        """Constructor function"""
        super(Data_Preparation, self).__init__()
        self.sequence_length = sequence_length
        self.dataset_length = dataset_length

    def integration_task(self, seq_len, num_samples):
        """Function which yields for num_samples random noise signals of the
        size seq_len.
        
        :param seq_len: length of sequence
        :type seq_len: integer
        :param num_len: number of datasamples
        :type seq_len: integer
        :return: input and target
        :rtype: tuple of Tensors
        """
        for _ in range(num_samples):
            x = tf.random.normal((seq_len,1))
            y = tf.expand_dims(tf.cast(tf.math.reduce_sum(x)>0, tf.int16), -1)
            yield x, y


    def my_integration_task(self):
        """Wrapper function that internally iterates through integration task
        
        :return: input and target
        :rtype: tuple of Tensors
        """
        for x,y in self.integration_task(self.sequence_length,
                                         self.dataset_length):
            yield x, y
    
    def split_dataset(self, ds, dataset_size=None, split_proportions = {
        'train': 0.7,
        'valid': 0.1,
        'test': 0.2}):
        """Split a tensorflow dataset into parts of a specified proportion.

        :param ds: tensorflow dataset to be splitted
        :type ds: tensorflow dataset
        :param dataset_size: size of the dataset that was provided, defaults to None
        :type dataset_size: integer, optional
        :param split_proportions: proportions to split the dataset, defaults to
            {'train': 0.7, 'valid': 0.1, 'test': 0.2}
        :type split_proportions: dict, keys are strings and values are floats
        :return: dictionaries of dataset splits
        :rtype: dictonary,  keys are strings and values are tf datasets
        """
        assert sum(split_proportions.values()) <= 1,\
            "The sum of split_proportions is larger than 1!"
        if not dataset_size:
            dataset_size = len(list(ds))
        split_dataset = {}
        for key, prop in split_proportions.items():
            samples = int(dataset_size * prop)
            split_dataset[key] = ds.take(samples)
            ds = ds.skip(samples)
        return split_dataset
    
    def preprocessing_pipeline(self, data, batch_size) -> tf.data:
        """Apply preproccesing pipeline to the given dataset.
        
        :param data: data to be preprocessed
        :type data: tensorflow 'Dataset'
        :param batch_size: batch size of the created dataset
        :type batch_size: integer
        :return: preprocessed dataset
        :rtype: tensorflow 'Dataset'
        """
        # cache the dataset
        data = data.cache()
        # shuffle, batch and prefetch the dataset
        data = data.shuffle(1000)
        data = data.batch(batch_size)
        data = data.prefetch(100)
        return data
    
    def generate_data(self, batch_size):
        """Generate data and return preprocessed datasplits
        
        :param batch_size: size of the batches
        :type batch_size: integer
        :return: dictionaries of dataset splits
        :rtype: dictonary, keys are strings and values are tf datasets
        """
        data = tf.data.Dataset.from_generator(
            self.my_integration_task,
            output_signature=(
                tf.TensorSpec(shape=(self.sequence_length,1), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.int16))
        )
        datasets = self.split_dataset(data, dataset_size=self.dataset_length)
        datasets = {key: self.preprocessing_pipeline(ds, batch_size)
                        for key, ds in datasets.items()}
        return datasets