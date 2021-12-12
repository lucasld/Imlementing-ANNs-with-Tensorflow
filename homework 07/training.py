import tensorflow as tf
import numpy as np


def train_step(model, input, target, loss_function, optimizer):
        """Apply optimizer to all trainable variables of a model to
        minimize the loss (loss_function) between the target output and the
        predicted ouptut.

        :param model: model to train
        :type mdoel: tensorflow model
        :param input: input to the model
        :type input: tf.Tensor
        :param target: target output with repect to the input
        :type target: tf.Tensor
        :param loss_function: loss function used to calculate loss of the model
        :type loss_function: function from the tf.keras.losses module
        :param optimizer: optimizer used to apply gradients to the models
            trainable variables
        :type optimizer: function from the tf.keras.optimizers module
        :return: the loss and the accuracy of the models prediction
        :rtype: tuple of two floats
        """
        with tf.GradientTape() as tape:
            prediction = model(input)
            loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        # apply gradients to the trainable variables using a optimizer
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        accuracy = arg_max_accuracy(prediction, target)
        return loss, accuracy


def test(model, test_data, loss_function):
    """Calculate the mean loss and accuracy of the model over all elements
    of test_data.

    :param model: model to train
    :type mdoel: tensorflow model
    :param test_data: model is evaulated for test_data
    :type test_data: tensorflow 'Dataset'
    :param loss_function: loss function used to calculate loss of the model
    :type loss_function: function from the tf.keras.losses module
    :return: mean loss and mean accuracy for all datapoints
    :rtype: tuple of two floats
    """
    # aggregator lists for tracking the loss and accuracy
    test_accuracy_agg = []
    test_loss_agg = []
    # iterate over all input-target pairs in test_data
    for (input, target) in test_data:
        prediction = model(input)
        loss = loss_function(target, prediction)
        accuracy = arg_max_accuracy(prediction, target)
        # add loss and accuracy to aggregators
        test_loss_agg.append(loss.numpy())
        test_accuracy_agg.append(np.mean(accuracy))
    # calculate mean loss and accuracy
    test_loss = tf.reduce_mean(test_loss_agg)
    test_accuracy = tf.reduce_mean(test_accuracy_agg)
    return test_loss, test_accuracy


def arg_max_accuracy(pred, target):
    """Calucalte accuracy between a prediction and a target.
    
    :param pred: a prediction that the model made
    :type pred: tf.Tensor of floats
    :param target: target that model should have predicted
    :type target: tf.Tensor of floats
    """
    #print("pred", pred,'target', target)
    #same_prediction = np.mean(target) == np.mean(pred)
    #same_prediction = np.argmax(target, axis=1) == np.argmax(pred, axis=1)
    same_prediction = np.round(target) == np.round(pred)
    return np.mean(same_prediction)


def training(input_model, datasets, loss_function, optimizer, epochs=10):
    """Training a model on a dataset for a certain amount of epochs.

    :param input_model: model to be trained
    :type input_model: model of type CustomModel
    :param datasets: train, validation and test datasets
    :type datasets: dictionary containing tf datasets keyed with: 'train',
        'test', 'valid'
    :param loss_function: loss function used to calculate loss of the model
    :type loss_function: function from the tf.keras.losses module
    :param optimizer: optimizer used to apply gradients to the models
        trainable variables
    :type optimizer: function from the tf.keras.optimizers module
    :param epochs: number of epochs to train on the dataset
    :type epochs: integer
    :return: losses and accuracies
    :rtype: tuple containing two dictonaries for the losses and accuracies.
        These are keyed like the dataset with 'train', 'valid' and 'test'
    """
    tf.keras.backend.clear_session()
    # Initialize lists for tracking loss and accuracy
    losses = {'train':[], 'valid':[], 'test':0}
    accuracies = {'train':[], 'valid':[], 'test':0}
        
    # Train-Dataset
    train_loss, train_accuracy = test(input_model, datasets['train'], loss_function)

    losses['train'].append(train_loss)
    accuracies['train'].append(train_accuracy)

    valid_loss, valid_accuracy = test(input_model, datasets['valid'], loss_function)
    #valid_losses.append(valid_loss)
    #valid_accuracies.append(valid_accuracy)
    losses['valid'].append(valid_loss)
    accuracies['valid'].append(valid_accuracy)

    # Training for epochs
    for epoch in range(1, epochs+1):
        last_valid_acc = np.round(accuracies['valid'][-1], 3)
        last_valid_loss = np.round(losses['valid'][-1], 3)
        print(f'Epoch {str(epoch)} starting with valid. accuracy: {last_valid_acc} and loss: {last_valid_loss}')
        epoch_loss_agg = []
        epoch_accuracy_agg = []
        for input, target in datasets['train']:
            train_loss, train_accuracy = train_step(
                input_model, input, target, loss_function, optimizer
            )
            epoch_loss_agg.append(train_loss)
            epoch_accuracy_agg.append(train_accuracy)
        # track training loss and accuracy
        losses['train'].append(tf.reduce_mean(epoch_loss_agg))
        accuracies['train'].append(tf.reduce_mean(epoch_accuracy_agg))
        # track loss and accuracy for test-dataset
        valid_loss, valid_accuracy = test(input_model, datasets['valid'], loss_function)
        losses['valid'].append(valid_loss)
        accuracies['valid'].append(valid_accuracy)
    test_loss, test_accuracy = test(input_model, datasets['test'], loss_function)

    losses['test'] = test_loss
    accuracies['test'] = test_accuracy    

    return losses, accuracies