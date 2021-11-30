from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


number_of_polys = 50
for zone_id in range(number_of_polys):

    zonaCount=0

    y = 0.006545
    x = 0.004579
    random_point = [(random.uniform(45.732526, 45.867707), random.uniform(15.817383670723778, 16.174692))]
    coords = [random_point[0],
              (random_point[0][0] + x, random_point[0][1]),
              (random_point[0][0] + x, random_point[0][1] - y),
              (random_point[0][0], random_point[0][1] - y)]

    #coords = [(45.8288, 15.8167), (45.7622, 15.8295), (45.7534, 16.1820), (45.8439, 16.1668)]
    #coords =  [(45.794573, 15.959467), (45.794378, 15.966065), (45.793832, 15.966172), (45.793877, 15.959070)]  #  zona dio slavonske
    print(coords)
    # random.uniform(45.732526, 45.939369) from 45.939369 to 45.732526771910415
    # random.uniform(15.817383670723778, 16.191989537997728) from 15.817383670723778 to 16.191989537997728
    zonaCount=zonaCount+1
    polygon = Polygon(coords)  # create polygon
    f = open("zagreb.txt", "r")
    lines = f.readlines()
    row =[]
    print("Zona" + str(zonaCount))
    for l in lines:
        row = l.split(";")
        point1 = Point(float(row[2]),float(row[1]))
        point2 = Point(float(row[4]),float(row[3]))
        if polygon.contains(point1) or polygon.contains(point2):
            print("Nasao sam! ID="+row[0])
            print("Link je tipa "+row[9])
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
                        print(profil)
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
                        np.save(f"result_data\\{row[0]}_{zone_id}", np_matrix)

                        # print(dataFrame)
                        #sns.heatmap(dataFrame)
                        #plt.show()
                else:
                    print("File is empty")
                    continue
            except OSError as e:
                print("Nastavak petlje")
                continue