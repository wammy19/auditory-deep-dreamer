import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import History


def plot_learning_metrics(
        history: History,
        metric: str,
        title: str,
        x_label: str,
        y_label: str,
        line_label: str,
        dots_label: str
) -> None:
    """
    :param: history: History returned from keras model tf.Model.fit() method.
    :param: title: Title for the graph.
    :param: x_label: x-axis label for the graph.
    :param: y_label: y-axis label for the graph.
    :param: metric: 'accuracy' or 'loss'
    :param: line_label: Label for the legend.
    :param: dots_label: Label for the legend.
    :return: None

    Plots a matplotlib graph for a keras.callbacks.History object that is tracking 'accuracy' and 'loss' metrics.
    """

    history_dict: dict = history.history
    plt.clf()  # Clear plot.

    if metric == 'loss':
        line = history_dict['val_loss']
        dots = history_dict['loss']

    else:
        line = history_dict['val_accuracy']
        dots = history_dict['accuracy']

    epochs = range(1, len(dots) + 1)
    blue_dots: str = 'bo'
    solid_red_line: str = 'r'

    plt.plot(epochs, line, solid_red_line, label=line_label)
    plt.plot(epochs, dots, blue_dots, label=dots_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_history(history: History) -> None:
    """
    :param: history: History returned from keras model tf.Model.fit() method.
    :return: None

    Plots accuracy and loss metrics.
    """

    # Train and validation loss.
    plot_learning_metrics(history, title='Training and validation loss', x_label='Epochs', y_label='Loss',
                          metric='loss', line_label='Validation loss', dots_label='Training loss')

    # Train and validation accuracy.
    plot_learning_metrics(history, title='Training and validation acc', x_label='Epochs', y_label='Accuracy',
                          metric='accuracy', line_label='Validation acc', dots_label='Training acc')


def display_optimal_epoch_and_metrics(_history: History) -> None:
    """
    :param: _history:
    :return: None

    Finds the most optimal epoch by finding the epoch at which the validation loss bottoms out.
    """

    optimal_epoch: int = np.array(_history.history['val_loss']).argmin()

    # Training.
    loss: float = round(_history.history['loss'][optimal_epoch], 2)
    acc: float = round(_history.history['accuracy'][optimal_epoch], 2)

    # Validation.
    val_loss: float = round(_history.history['val_loss'][optimal_epoch], 2)
    val_acc: float = round(_history.history['val_accuracy'][optimal_epoch], 2)

    print(f'\nOptimal Epoch: {optimal_epoch + 1}')
    print(f'Training Loss: {loss:.5f}        |  Training Accuracy: {acc:.5f}')
    print(f'Validation Loss: {val_loss:.5f}  |  Validation Accuracy: {val_acc:.5f}\n')
