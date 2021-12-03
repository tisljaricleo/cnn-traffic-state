from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
import pickle


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


def get_label(data_matrix, congestion_speed):
    label = 0
    speed_count = (data_matrix < congestion_speed).sum()

    if speed_count >= 60:
        label = 1

    for i in range(data_matrix.shape[0]):
        row_sum = (data_matrix[i, :] < congestion_speed).sum()
        if row_sum >= 5 and speed_count >= 60:
            label = 2

    return label


number_of_polys = 20000
poly_coounter = 0
result_list = []

for zone_id in range(number_of_polys):

    poly_coounter += 1
    print(f"Poly counter:{poly_coounter}/{number_of_polys}")

    zonaCount=0

    y = 0.006545
    x = 0.004579
    random_point = [(random.uniform(45.732526, 45.867707), random.uniform(15.817383670723778, 16.174692))]
    coords = [random_point[0],
              (random_point[0][0] + x, random_point[0][1]),
              (random_point[0][0] + x, random_point[0][1] - y),
              (random_point[0][0], random_point[0][1] - y)]

    # print(coords)

    zonaCount=zonaCount+1
    polygon = Polygon(coords)  # create polygon
    f = open("zagreb.txt", "r")
    lines = f.readlines()
    row =[]
    # print("Zona" + str(zonaCount))

    for l in lines:
        row = l.split(";")
        point1 = Point(float(row[2]),float(row[1]))
        point2 = Point(float(row[4]),float(row[3]))
        if polygon.contains(point1) or polygon.contains(point2):
            # print("Nasao sam! ID="+row[0])
            # print("Link je tipa "+row[9])
            div1 = 10000
            id = int(row[0])
            b = int(id / div1)
            path1 = str(b * div1) + "-" + str((b * div1) + div1)
            mod = id % div1
            div2 = 100
            c = int(mod / div2)
            path2 = str((b * div1) + (c * div2)) + "-" + str(((b * div1) + (c * div2) + div2))

            data = os.path.abspath(
                "D:/DATA_/samo_profili/SP_24_06_Whole/SPPerLinks/" + path1 + "/" + path2 + "/" + str(id) + "/SPInfo.txt")
            #print(data)
            try:
                if os.path.getsize(data) > 0:
                    file1 = open(data, 'r')
                    file2 = open(data, 'r')
                    file3 = open(data, 'r')
                    Lines = file1.readlines()[3]
                    Days = file2.readlines()[1]
                    ProfilTxt = file3.readlines()[6]
                    name = []
                    day =[]
                    pro =[]
                    name = Lines.split()
                    day = Days.split()
                    profil =ProfilTxt.split()
                    if len(profil) > 14:
                        #print(name)
                        #print(day)
                        #print(profil)
                        profil.pop(0)
                        # print(profil)
                        dataFrame = pd.DataFrame()
                        for index in range(len(profil)):
                            l = []
                            l = profil[index].split("_")
                            if l[2] == "SP":
                                putanja = os.path.abspath("D:/DATA_/samo_profili/SP_24_06_Whole/SPPerLinks/" + path1 + "/" + path2 + "/" + str(id) + "/" + profil[index])
                                df = pd.read_csv(putanja, delimiter = "\t")
                                """Matrica 198 x 14 jer su 5 min intevali od 5:30 do 22:00 a 14 jer su to brzine za oba smjera"""
                                dataFrame[str(index)]=df.REL

                        np_matrix = dataFrame.to_numpy()

                        np_matrix_dir_1 = np_matrix[:, :7]
                        np_matrix_dir_2 = np_matrix[:, 7:]

                        c_speed = 50
                        # print("LABELS:")
                        label_1 = get_label(np_matrix_dir_1, c_speed)
                        label_2 = get_label(np_matrix_dir_2, c_speed)

                        result_list.append(
                            {f"{row[0]}_{zone_id}_1": np_matrix_dir_1,
                             "label": label_1
                             })

                        result_list.append(
                            {f"{row[0]}_{zone_id}_2": np_matrix_dir_2,
                             "label": label_2
                             })


                        # Naming: linkid_zoneid_direction
                        # np.save(f"result_data\\{row[0]}_{zone_id}_1", np_matrix_dir_1)
                        # np.save(f"result_data\\{row[0]}_{zone_id}_2", np_matrix_dir_2)


                        # sns.heatmap(np_matrix_dir_1)
                        # plt.show()
                        # sns.heatmap(np_matrix_dir_2)
                        # plt.show()

                        # print(dataFrame)
                        #sns.heatmap(dataFrame)
                        #plt.show()
                else:
                    print("File is empty")
                    continue
            except OSError as e:
                # print("Nastavak petlje")
                continue

save_pickle_data("result_data/result.pkl", result_list)
