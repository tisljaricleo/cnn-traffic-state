

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import numpy as np


def get_data_paths(data_dir: str):
    dp = []
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            dp.append(os.path.join(data_dir, file))
    return dp


data_paths = get_data_paths("result_data")

tensor_data = np.zeros((len(data_paths), 198, 14, 3))
labels = np.zeros((len(data_paths), 1))

for i in range(len(data_paths)):
    matrix = np.load(data_paths[i])

    speed_count = (matrix < 50).sum()
    if speed_count > 60:
        labels[i, 0] = 1

    matrix_3d = np.repeat(matrix[:, :, np.newaxis], 3, axis=2)
    tensor_data[i, :, :, :] = matrix_3d


# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#
# # Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0
#
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']
#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(198, 14, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(16, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
#
# model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2))
print(model.summary())


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(tensor_data, labels, epochs=10, verbose=True)


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
#
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#
# print(test_acc)


