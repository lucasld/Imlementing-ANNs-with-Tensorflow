import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import RangeSlider


def show_images(datasets, normalize=False, shape=(3,5), show_title=True):
    r,c = shape
    fig, axs = plt.subplots(r, c)
    fig.set_size_inches(20,10)

    for img_batch, label_batch in datasets['test'].take(1):
        for y,x in np.ndindex((r, c)):
            img = np.array(img_batch[y*5+x])
            if normalize:
                img_max = img.max()
                img_min = img.min()
                img = (img-img_min)/(img_max-img_min)
            if show_title: axs[y][x].set_title(np.argmax(label_batch[y*5+x]))
            axs[y][x].axis('off')
            axs[y][x].imshow(img)
    plt.show()


def plot_results(losses, accuracies):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(20, 6)

    fig.suptitle('Training Progress for the Fashion-MNIST Dataset')
    axs[0].plot(losses['train'], color='orange', label='train losses')
    axs[0].plot(losses['valid'], color='green', label='validation losses')
    axs[0].axhline(y=losses['test'], color='r', linestyle='-', label='test loss')
    axs[0].set(ylabel='Losses')
    axs[0].legend()
    axs[1].plot(accuracies['train'], color='orange', label='train accuracies')
    axs[1].plot(accuracies['valid'], color='green', label='validation accuracies')
    axs[1].axhline(y=accuracies['test'], color='r', linestyle='-', label='test accuracy')
    axs[1].set_yticks(np.arange(0,1.1,0.1))
    axs[1].set(xlabel='Epochs', ylabel='Accuracies')
    axs[1].legend()
    

def show_layer_output(model, input):
    import ipywidgets as widgets
    outputs = []
    x = input
    for i, layer in enumerate(model.layer_list):
        #axs[i].imshow(x[0,:,:,0])
        if type(layer) is tf.keras.layers.GlobalAveragePooling2D: break#
        x = layer(x)
        outputs.append(x)
    # Create the RangeSlider
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Threshold",0,10)
    current_channel = 0
    fig, axs = plt.subplots(1, len(outputs))
    fig.set_size_inches(20,6)
    for i,o in enumerate(outputs):
        axs[i].imshow(o[0,:,:,current_channel])
    
    @widgets.interact(w=(0, 10, 1), amp=(0, 4, .1), phi=(0, 2*np.pi+0.01, 0.01))
    def update():
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        # Update the image's colormap
        
        for i,o in enumerate(outputs):
            axs[i].imshow(o[0,:,:,np.random.randint(0,10)])
            
    def say_my_name(name):
        """
        Print the current widget value in short sentence
        """
        print(f'My name is {name}')
        
    widgets.interact(say_my_name, name=["Jim", "Emma", "Bond"])

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()
    slider.on_changed(update)
    
