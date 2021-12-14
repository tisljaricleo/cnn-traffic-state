import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


def save_pickle_data(path, data):
    """Saves data in the pickle format
    :param path: Path to save
    :param data: Data to save
    :type path: str
    :type data: optional
    :return:
    """
    try:
        with open(path, 'wb') as handler:
            pickle.dump(data, handler, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)


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


def plot_acc_curve(model_history, plot_type: str, plot_name="acc_epoch.png"):
    print("Plotting started ...")
    plt.plot(model_history.history['accuracy'], label='accuracy')
    plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.2])
    plt.legend()
    if plot_type == "show":
        plt.show()
    else:
        plt.savefig(f"plots/{plot_name}")


def get_data_paths(data_dir: str):
    dp = []
    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            dp.append(os.path.join(data_dir, file))
    return dp


data_paths = get_data_paths("result_data")
data_x = open_pickle(data_paths[0])

print("Data loaded!")

# Initialize empty tensor and labels arrays.
tensor_data = np.zeros((len(data_x), 198, 7, 3))
labels = np.zeros((len(data_x), 1))

# Counter for invalid input images.
invalid_counter = 0

# Create a tensor for training and test.
# CNN input tensor is (n_instances, height, width, 3).
for i in range(len(data_x)):
    matrix_name = list(data_x[i].keys())[0]
    label_name = list(data_x[i].keys())[1]

    if matrix_name == "label":
        print("Wrong matrix name!!! " * 10)

    matrix = data_x[i][matrix_name]
    label = data_x[i][label_name]

    # Check the matrix shape and skip invalid shapes.
    shp = matrix.shape
    if not (shp[0] == 198 and shp[1] == 7):
        invalid_counter += 1
        continue

    labels[i, 0] = label
    matrix_3d = np.repeat(matrix[:, :, np.newaxis], 3, axis=2)
    tensor_data[i, :, :, :] = matrix_3d

print(f"Invalid counter:{invalid_counter}")
print("Data split to tensor data and labels!")

# Initialize the CNN model setup.
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(198, 7, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3))
print(model.summary())
print("Model created!")

# Compile the model.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', 'mae', 'mape'])
print("Model compiled!")

# Split the initial tensor to train and test data.
X_train, X_test, y_train, y_test = train_test_split(tensor_data,
                                                    labels,
                                                    test_size=0.33,
                                                    random_state=42)

# save_pickle_data("X_test.pkl", X_test)
# save_pickle_data("y_test.pkl", y_test)

print("Data is split to train and test!")

print("Learning started!")
history = model.fit(X_train,
                    y_train,
                    epochs=50,
                    verbose=True,
                    validation_data=(X_test, y_test))
print("Learning finished!")

# Show the final model evaluation results.
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

# Plot the acc curve to see if the model should be saved.
plot_acc_curve(history, "show")
print("Plotting finished!")

# Saving result accuracy curves to file.
# User input required because sometimes the model is not performing well.
# Then the results should not be saved.
print("Do you want to save accuracy curves to file? (y/n)")
response = input()
if response in ["y", "Y"]:
    print("Plot name and format to save: (def: acc_epoch.png)")
    response_name = input()
    if response_name == "":
        plot_acc_curve(history, "save")
    else:
        plot_acc_curve(history, "save", response_name)
    print("Saving finished!")

# Saving result model and training history to file.
# User input required because sometimes the model is not performing well.
# Then the results should not be saved.
print("Do you want to save training history and model? (y/n)")
response_save = input()
if response_save in ["y", "Y"]:
    print("Model name and format to save: (def: final_model.h5)")
    response_name = input()
    if response_name == "":
        save_pickle_data("final_model/training_history_final_model.pkl", history.history)
        model.save("final_model/final_model.h5")
    else:
        save_pickle_data(f"final_model/training_history_{response_name}.pkl", history.history)
        model.save(f"final_model/{response_name}")



