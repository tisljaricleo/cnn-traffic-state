import cv2 as cv
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sns

# np.set_printoptions(threshold=sys.maxsize, linewidth=1000)


def get_data_paths(data_dir: str):
    dp = []
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            dp.append(os.path.join(data_dir, file))
    return dp


def plot_one_matrix(matrices, interval, first_link, second_link):
    """
    Plots the heatmap for one matrix.
    :param matrices: (DataFrame) Pandas DataFrame with all matrices.Spatial matrix.
    :param interval: (int) Interval index.
    :param first_link: (string) First link (origin) index name. (ex. '1_4')
    :param second_link: (string) Second link (destination) index name. (ex. '4_1')
    :return:
    """
    out = np.array(matrices[interval - 1][second_link][first_link])
    out = np.squeeze(out, axis=0)
    return out


def plot_heatmap_nosave(data):
    """
    Plots heatmap WITHOUT saving to png format.
    :param data: Any 2D numpy array.
    :return:
    """
    # states_names = list(range(10, 110, 10))
    # title = 'Count matrix'

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # We want to show all ticks...
    # ax.set_xticks(np.arange(len(states_names)))
    # ax.set_yticks(np.arange(len(states_names)))
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(states_names)
    # ax.set_yticklabels(states_names)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(states_names)):
    #     for j in range(len(states_names)):
    #         text = ax.text(j, i, data[i, j],
    #                        ha="center", va="center", color="w")

    # ax.set_title(title)
    # fig.tight_layout()
    plt.show()


def get_sparsity(m):
    # m = plot_one_matrix(matrices, 5, '1_4', '4_5')
    # m = plot_one_matrix(matrices, 5, '23_18', '18_17')

    # img1 = cv.imread('j.png',0)
    # img = m

    m = np.where(m >= 70, 100, m)

    max_val = m.max()
    factor = 255 / max_val
    m_fact = m * factor
    m_fact = m_fact.astype(np.uint8)

    img = m_fact
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv.erode(img, kernel, iterations=1)
    canny = cv.Canny(erosion, 10, 10)
    # erosion = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    sns.heatmap(m)
    plt.show()
    sns.heatmap(m_fact)
    plt.show()
    sns.heatmap(erosion)
    plt.show()
    sns.heatmap(canny)
    plt.show()


    # plot_heatmap_nosave(m)
    # plot_heatmap_nosave(m_fact)
    # plot_heatmap_nosave(erosion)
    # plt.imshow(canny, cmap='gray')

    print("---------original---------------")
    m_count_nonzero = np.count_nonzero(m)
    print(100 - m_count_nonzero)
    print((100 - m_count_nonzero) / 100)
    print("---------factor---------------")
    m_fact_count_nonzero = np.count_nonzero(m_fact)
    print(100 - m_fact_count_nonzero)
    print((100 - m_fact_count_nonzero) / 100)
    print("---------erosion---------------")
    m_erosion_count_nonzero = np.count_nonzero(erosion)
    print(100 - m_erosion_count_nonzero)
    print((100 - m_erosion_count_nonzero) / 100)


paths = get_data_paths("result_data")

for path in paths:
    x = np.load(path)

    get_sparsity(x)

# max_val = m.max()
# factor = 255/max_val
# m_fact = m*factor
# m_fact = m_fact.astype(np.uint8)
#
# img = m_fact
# kernel = np.ones((2,2),np.uint8)
# erosion = cv.erode(img,kernel,iterations = 1)
#
# plot_heatmap_nosave(m)
# plot_heatmap_nosave(m_fact)
# plot_heatmap_nosave(erosion)
#
# print("---------original---------------")
# m_count_nonzero = np.count_nonzero(m)
# print(100 - m_count_nonzero)
# print((100 - m_count_nonzero)/100)
# print("---------factor---------------")
# m_fact_count_nonzero = np.count_nonzero(m_fact)
# print(100 - m_fact_count_nonzero)
# print((100 - m_fact_count_nonzero)/100)
# print("---------erosion---------------")
# m_erosion_count_nonzero = np.count_nonzero(erosion)
# print(100 - m_erosion_count_nonzero)
# print((100 - m_erosion_count_nonzero)/100)

