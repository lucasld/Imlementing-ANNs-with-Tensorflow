import matplotlib.pyplot as plt
import numpy as np

def show_dataset_examples(dataset, shape=(1, 5), normalize=True) -> None:
    r,c = shape
    fig, axs = plt.subplots(r*2, c)
    fig.set_size_inches(20,10)

    for img_batch, label_batch in dataset.take(1):
        for y,x in np.ndindex((r, c)):
            input_img = np.array(img_batch[y*c+x])
            target_img = np.array(label_batch[y*c+x])
            if normalize:
                img_max = input_img.max()
                img_min = input_img.min()
                input_img = (input_img-img_min)/(img_max-img_min)
                img_max = target_img.max()
                img_min = target_img.min()
                target_img = (target_img-img_min)/(img_max-img_min)
            axs[y*2][x].axis('off')
            axs[y*2][x].imshow(input_img[:,:,0])
            axs[y*2+1][x].axis('off')
            axs[y*2+1][x].imshow(target_img[:,:,0])

    plt.show()


def plot_results(losses, accuracies, title) -> None:
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