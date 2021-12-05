import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def load_cifar10(split_list=['train[:85%]', 'train[85%:]', 'test']) -> dict:
    """Load the CIFAR-10 dataset and split it into train, validation and
    testdatasets.

    :param split_list: a list containing proportions to split the test and/or
        train datasets to produce a validation dataset,
        dafaults to: ['train[:85%]', 'train[85%:]', 'test']
    :type split_list: a list of 3 strings in tensorflow split-notation,
        optional
    :return: A dictonary containing 3 tensorflow datasets keyed by 'train',
        'valid' and 'test'
    :rtype: dictonary containing tensorflow datasets
    """
    assert len(split_list) == 3, 'The split list should contain 3 strings!'
    (train_ds, valid_ds, test_ds), fashion_mnist_info = tfds.load(
        'cifar10',
        split=split_list,
        as_supervised=True,
        with_info=True,
        shuffle_files=True)
    return {'train':train_ds, 'valid':valid_ds, 'test':test_ds}


def compute_mean_std_tensor(dataset): #-> tuple[np.ndarray, np.ndarray]
    """Compute the mean and the standard deviation for every feature/pixel over
    the images of the given dataset.

    :param dataset: a tensorflow dataset for which mean and standart deviations
        should be calculated
    :type dataset: tensorflow dataset
    :return: mean and standart deviation arrays, same shape as the input images
        in the dataet (32,32,3)
    :rtype: tuple of two numpy arrays
    """
    # extract all imgages from the dataset and write them into an numpy array
    np_imgs = np.array([img for img, _ in tfds.as_numpy(dataset)])
    # calculate mean for each pixel over all images
    np_means = np.empty(np_imgs.shape[1:])
    np_std = np.empty(np_imgs.shape[1:])
    for y, x, channel in np.ndindex(np_means.shape):
        # add mean and std for specific pixel location
        np_means[y,x] = np.mean(np_imgs[:,y,x, channel])
        np_std[y,x] = np.std(np_imgs[:,y,x, channel])
    return np_means, np_std


def preprocessing_pipeline(data, means_array, std_array, batch_size) -> tf.data:
    """Apply preproccesing pipeline to the given dataset.
    
    :param data: data to be preprocessed
    :type data: tensorflow 'Dataset'
    :param means_array: array of the same shape as an image, containing every
        feature's mean over all images in the train dataset
    :type means_array: numpy array of floats
    :param std_array: array of the same shape as an image, containing every
        feature's standart deviation over all images in the train dataset
    :type std_array: numpy array of floats
    :param batch_size: batch size of the created dataset
    :type batch_size: integer
    :return: preprocessed dataset
    :rtype: tensorflow 'Dataset'
    """
    # standartize images and one-hot targets
    data = data.map(lambda image, target: (
        (tf.cast(image, tf.float32) - means_array) / std_array,
        tf.one_hot(int(target), 10))
    )
    # cache the dataset
    data = data.cache()
    # shuffle, batch and prefetch the dataset
    data = data.shuffle(1000)
    data = data.batch(batch_size)
    data = data.prefetch(100)
    return data


def create_datasets(batch_size=128) -> dict:
    """Load, split and preprocess the CIFAR-10 Dataset.
    
    :param batch_size: batch size of the created dataset, defaults to 64
    :type batch_size: integer, optional
    :return: dictonary containing 3 datasets keyed by 'train', 'valid' and
        'test'
    :rtype: dictonary of tensorflow datasets
    """
    datasets = load_cifar10()
    img_means, img_stds = compute_mean_std_tensor(datasets['train'])
    datasets = {key:preprocessing_pipeline(ds, img_means, img_stds, batch_size)
                for key, ds in datasets.items()}
    return datasets
