import matplotlib.pyplot as plt
import pickle
import tensorflow as tf


def plot_acc_curve(model_history, plot_type: str, plot_name="acc_epoch.png"):
    print("Plotting started ...")
    plt.plot(model_history['accuracy'], label='accuracy')
    plt.plot(model_history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.2])
    plt.legend()
    if plot_type == "show":
        plt.show()
    else:
        plt.savefig(f"plots/{plot_name}")


def open_pickle(path):
    """Opens pickle data from defined path
    :param path: Path to pickle file
    :type path: str
    :return:
    """
    try:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
            return data
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
            return None
        else:
            print(e)
            return None


model_name = "final_model.h5"
history_name = "training_history_final_model.pkl"

X_test = open_pickle("X_test.pkl")
y_test = open_pickle("y_test.pkl")

history = open_pickle(f"final_model/{history_name}")
new_model = tf.keras.models.load_model(f"final_model/{model_name}")
new_model.summary()

plot_acc_curve(history, "save", model_name[:-3])

# Show the final model evaluation results.
test_loss, test_acc = new_model.evaluate(X_test,  y_test, verbose=2)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

print()

