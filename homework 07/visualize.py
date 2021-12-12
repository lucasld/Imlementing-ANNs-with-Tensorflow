import matplotlib.pyplot as plt


def plot_results(losses, accuracies, title):
    """Plot losses and accuracies.

    :param losses: dict of losses. keys are exypected to be train, valid
        and test
    :type losses: dictionary of lists
    :param accuracies: dict of accuracies. keys are exypected to be train,
        valid and test
    :type accuracies: dictionary of lists
    :param title: title of the plot
    :type title: string
    """
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(20, 6)

    fig.suptitle(title)
    axs[0].plot(losses['train'], marker='.', color='orange', label='train losses')
    axs[0].plot(losses['valid'], marker='.', color='green', label='validation losses')
    axs[0].axhline(y=losses['test'], color='blue', linestyle='--', label='test loss')
    axs[0].set(ylabel='Losses')
    axs[0].legend()
    axs[1].plot(accuracies['train'], marker='.', color='orange', label='train accuracies')
    axs[1].plot(accuracies['valid'], marker='.', color='green', label='validation accuracies')
    axs[1].set(xlabel='Epochs', ylabel='Accuracies')
    axs[1].axhline(y=accuracies['test'], color='blue', linestyle='--', label='test accuracy')
    axs[1].axhline(y=[0.8], color='r', linestyle='--', label='expected accuracy')
    axs[1].legend()

    axs[0].grid()
    axs[1].grid()
    plt.show()