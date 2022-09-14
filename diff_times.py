import numpy
import subprocess
import os
import json
import time
import sys
import random


def comp_times(task, dev1, dev2, dev3):
    '''#             Exo(Huge) Exo(small) Exo(med) Exo(lg)  EGS  Lenovo  NJN   RPi4   RPi3
    encode_200 = [0.23,0,0,0.44, 0.4, 0.27, 0.17, 0.33, 1.9, 2.16, 2.5]  # seconds
    encode_1500 = [0.43,0,0,0.65, 0.62, 0.5, 0.36, 0.42, 2.63, 3.19, 7.35]  # seconds
    encode_3000 = [0.55,0,0,0.9, 0.89, 0.65, 0.47, 0.59, 3.48, 4.4, 8.44]  # seconds
    encode_6500 = [1.3,0,0,2.7, 2.58, 1.43, 1.22, 1.59, 9.68, 11.8, 22.7]  # seconds
    encode_20000 = [2.6,0.0,0.0,6.2, 6.1, 3.1, 2.7, 3.16, 20.64, 28, 60]  # seconds

    #           Exo(Huge)  Exo(small) Exo(med) Exo(lg) EGS  Lenovo  NJN   RPi4   RPi3
    frame_200 = [0.77,0,0,0.84, 0.8, 0.5, 0.5, 0.6, 2, 4, 11]
    frame_1500 = [1.83,0,0,2.2, 1.8, 2, 2.5, 2, 9.4, 11, 20]
    frame_3000 = [2,0,0,2.9, 2.5, 3, 3.7, 3.1, 14, 14, 26]
    frame_6500 = [8.4,0,0,9, 8.7, 9, 14, 13.5, 55, 49, 88]
    frame_20000 = [17.5,0,0,21, 18.5, 18, 31, 31, 117, 112, 204]

    # 		    Exo(Huge)  Exo(small) Exo(med) Exo(lg)     EGS  Lenovo  NJN   RPi4   RPi3
    inference = [0.246,0,0,0.29, 0.26, 0.25, 0.23, 0.28, 1.94, 1.1, 1.5]
    lowtrain = [19.6,0,0,32, 26.5, 25.76, 17, 18, 152, 102, 1000]  # seconds
    hightrain = [62.5,0,0,109, 103, 70.6, 33, 57, 232, 467, 1000]  # seconds'''

    #      V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
    encode_200 = [0.4, 0.44, 0.46, 0.4, 0.4, 0.28, 0.17, 0.33, 1.9, 2.16, 0.23]  # seconds
    encode_1500 = [0.62, 0.65, 1.14, 0.62, 0.62, 0.5, 0.36, 0.42, 2.63, 3.19, 0.43]  # seconds
    encode_3000 = [0.89, 0.9, 1.6, 0.89, 0.89, 0.63, 0.47, 0.59, 3.48, 4.4, 0.55]  # seconds
    encode_6500 = [2.58, 2.7, 4.9, 2.58, 2.58, 1.45, 1.22, 1.59, 9.68, 11.8, 1.3]  # seconds
    encode_20000 = [6.1, 6.2, 11, 6.1, 6.1, 3.1,2.7, 3.16, 20.64, 28, 2.6]  # seconds

    #           V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
    frame_200 = [0.42,0.44,0.5,0.8, 0.8, 0.5,0.39,0.35, 2, 4, 0.77]
    frame_1500 = [1.7,1.8,2,1.8, 2,2.46,2.1, 9.4, 11, 1.83]
    frame_3000 = [2.5,2.6,2.8, 2.5, 2.5, 2.9,3.8,3.2, 14, 14, 2]
    frame_6500 = [8.6,9.4,9.8,8.7, 8.7, 9.9,14.2,11.8, 55, 49, 8.4]
    frame_20000 = [18,18.2,21,18.5,18.5, 20.9,31,26, 117, 112, 17.5]

    # 		   V-Exo(med) V-Exo(small) V-Exo(tiny)  Z-Exo(med) M-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
    inference = [0.26,0.29,0.3,0.26, 0.28,0.225,0.282, 0.28, 1.94, 1.1, 0.246]
    lowtrain = [26.5,32,35,26.5, 26.5, 23,17,18, 152, 102, 19.6]  # seconds
    hightrain = [103,109,115,103, 103, 77,33,57, 232, 467, 62.5]  # seconds

    # https://aws.amazon.com/kinesis/data-firehose/pricing/?nc=sn&loc=3

    # seg_size = 80000 #(10KB)
    # video_size = 8000000000 #(1GB)
    index_of_segment = 4
    seg_size = [286720, 2457600, 3440640, 14400000, 20971520]  # bits
    video_size = [2000000, 14000000, 28000000, 60000000, 204800000]  # bits

    # tasks0 = sys.argv[1].strip("][").split(",")
    ####print(tasks0)

    # resources = sys.argv[3].strip("][").split(",")

    # Converting string to list
    tasks0 = ["encode_20000", "frame_20000", "lowtrain", "hightrain", "inference", "transcode", "package"]  # sys.argv[1].strip("][").split(",")
    # print(type(tasks0))

    num_of_apps = 1
    newtasks = [0 for i in range(len(tasks0) * num_of_apps)]
    for k in range(num_of_apps):
        for i in range(len(tasks0)):
            newtasks[i + (k * len(tasks0))] = tasks0[i] + str(k)

    #              0	       1	      2      		3      4        5        6       7		8      9      10
    resources = ["V-Exo(med)", "V-Exo(small)", "V-Exo(tiny)",  "Z-Exo(med)", "M-Exo(med)", "Edge-kla", "AAu(large)", "Lenovo",  "NJN",   "RPi4",   "V-Exo(med)"]
    # resources = sys.argv[2].strip("][").split(",")

    lat = [[ 0.5e-3,0.5e-3,0.5e-3,12.5e-3,7.3e-3,4.8e-3,7.2e-3,7.2e-3,7.5e-3,7.5e-3,15e-3],
                [0.5e-3,0.5e-3,0.5e-3,12.5e-3,7.3e-3,4.8e-3,7.2e-3,7.2e-3,7.5e-3,7.5e-3,15e-3],
                [0.5e-3, 0.5e-3, 0.5e-3,12.5e-3, 7.3e-3, 4.8e-3, 7.2e-3, 7.2e-3, 7.5e-3, 7.5e-3, 15e-3],
                [12.5e-3,12.5e-3,12.5e-3,0.5e-3,6.7e-3,16.6e-3,23.2e-3,23.6e-3,23.2e-3,23.6e-3,25.9e-3],
                [7.3e-3,7.3e-3,7.3e-3,6.7e-3,0.5e-3,11.5e-3,12.2e-3,12.5e-3,12.6e-3,12.6e-3,21e-3],
                [4.8e-3,4.8e-3,4.8e-3,16.6e-3,11.5e-3,0.5e-3,11.4e-3,11.5e-3,12e-3,11.5e-3,10e-3],
                [7.2e-3,7.2e-3,7.2e-3,23.2e-3,12.2e-3,11.4e-3,0.5e-3,0.5e-3,0.5e-3,0.5e-3,22.4e-3],
                [7.2e-3,7.2e-3,7.2e-3,23.6e-3,12.5e-3,11.5e-3,0.5e-3,0.5e-3,0.5e-3,0.5e-3,22.8e-3],
                [7.5e-3,7.5e-3,7.5e-3,23.2e-3,12.6e-3,12e-3,0.5e-3,0.5e-3,0.5e-3,0.5e-3, 22.8e-3],
                [7.5e-3, 7.5e-3, 7.5e-3, 23.2e-3, 12.6e-3, 12e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3,22.6e-3],
                [15e-3,15e-3,15e-3,25.9e-3,21e-3,10e-3,22.4e-3,22.8e-3, 22.8e-3,22.6e-3,0.5e-3]]

    SIZE = 208
    BW_r = [[13000,13000,13000,1950, 3000, 5000,950, 900,900,900,1500],
           [13000,13000,13000,1950,3000,5000, 950,900,900,900,1500],
           [13000,13000,13000,1950,3000,5000,950,900,900,900,1500],
           [1950,1950,1950,1200,3200,1400,900,850,700,770, 900],
           [3000,3000,3000,3200,1200,2100,930,900,900, 850,1100],
           [5000,5000,5000,1400,2100,12000,930,860,840,850,1200],
           [950,950,950,900,930,930,930,930,930,850,920],
           [900,900,900,850,900,860,930,860,920,850,900],
           [900,900,900,700,900,840,930,850,920,920,900],
           [900,900,900,770,850,850,850,850,920,850,850],
           [1500,1500,1500, 900, 1100,1200, 920,900,900, 850,12000]]

    """
    lat = [[0.4e-3, 26e-3, 15e-3, 22.4e-3, 22.8e-3, 22.8e-3, 22.8e-3, 22.8e-3],
           [26e-3, 0.5e-3, 12.5e-3, 18.4e-3, 18.4e-3, 18.4e-3, 18.4e-3, 18.4e-3],
           [15e-3, 12.5e-3, 0.5e-3, 7.2e-3, 7.2e-3, 7.5e-3, 7.5e-3, 7.5e-3],
           [22.4e-3, 18e-3, 7.2e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3],
           [22.8e-3, 18.4e-3, 7.2e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3],
           [22.8e-3, 18.4e-3, 7.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3],
           [22.8e-3, 18.4e-3, 7.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3],
           [22.8e-3, 18.4e-3, 7.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3]]

    SIZE = 208

    # BW_r = [100000000 , 110000000, 220000000, 920000000, 920000000, 850000000 , 800000000 , 328000000]#bps
    BW_r = [[12, 0.76, 1.5, 0.92, 0.9, 0.9, 0.85, 0.4],
            [0.76, 12, 1.6, 0.93, 0.85, 0.7, 0.77, 0.4],
            [1.5, 1.6, 13, 0.95, 0.9, 0.9, 0.9, 0.4],
            [0.92, 0.93, 0.95, 0.9, 0.86, 0.93, 0.85, 0.4],
            [0.9, 0.85, 0.9, 0.86, 0.9, 0.92, 0.85, 0.4],
            [0.9, 0.7, 0.9, 0.93, 0.92, 0.9, 0.88, 0.4],
            [0.85, 0.77, 0.9, 0.85, 0.85, 0.88, 0.9, 0.4],
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]
  
"""

    T = [0 for i in range(len(resources))]
    Tm = [0 for i in range(len(resources))]
    Tr = [0 for i in range(len(resources))]

    # print (type(Tm))
    # print (type(Tr))
    if (task == "encode_20000"):
        ###print(encode_20000[0]," ",type(rec))
        Tm[0] = encode_20000[0]  # data size: 8sec video.
        Tr[0] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0])        
        T[0] = numpy.round(numpy.round(Tm[0], 4) +  numpy.round(Tr[0], 4), 4)
        #print (T[0])

        Tm[1] = encode_20000[1]
        Tr[1] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][1]) + (lat[dev1][1])        
        T[1] = numpy.round(numpy.round(Tm[1], 4) +  numpy.round(Tr[1], 4), 4)
        #print (T[1])

        Tm[2] = encode_20000[2]
        Tr[2] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][2]) + (lat[dev1][2])        
        T[2] = numpy.round(numpy.round(Tm[2], 4) +  numpy.round(Tr[2], 4), 4)
        #print (T[2])

        Tm[3] = encode_20000[3]
        Tr[3] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][3]) + (lat[dev1][3])        
        T[3] = numpy.round(numpy.round(Tm[3], 4) +  numpy.round(Tr[3], 4), 4)
        #print (T[3])

        Tm[4] = encode_20000[4]
        Tr[4] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][4]) + (lat[dev1][4])        
        T[4] = numpy.round(numpy.round(Tm[4], 4) +  numpy.round(Tr[4], 4), 4)
        #print (T[4])

        Tm[5] = encode_20000[5]
        Tr[5] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][5]) + (lat[dev1][5])        
        T[5] = numpy.round(numpy.round(Tm[5], 4) +  numpy.round(Tr[5], 4), 4)
        #print (T[5])

        Tm[6] = encode_20000[6]
        Tr[6] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][6]) + (lat[dev1][6])        
        T[6] = numpy.round(numpy.round(Tm[6], 4) +  numpy.round(Tr[6], 4), 4)
        #print (T[6])

        Tm[7] = encode_20000[7]
        Tr[7] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][7]) + (lat[dev1][7])        
        T[7] = numpy.round(numpy.round(Tm[7], 4) +  numpy.round(Tr[7], 4), 4)

        Tm[8] = encode_20000[8]
        Tr[8] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][8]) + (lat[dev1][8])        
        T[8] = numpy.round(numpy.round(Tm[8], 4) +  numpy.round(Tr[8], 4), 4)


        Tm[9] = encode_20000[9]
        Tr[9] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][9]) + (lat[dev1][9])        
        T[9] = numpy.round(numpy.round(Tm[9], 4) + numpy.round(Tr[9], 4), 4)


        Tm[10] = encode_20000[10]
        Tr[10] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][10]) + (lat[dev1][10])        
        T[10] = numpy.round(numpy.round(Tm[10], 4) +  numpy.round(Tr[10], 4), 4)
        # print (T[10])

    elif (task == "frame_20000"):
        ###print(frame_20000[0]," ",type(rec))
        Tm[0] = frame_20000[0]  # data size: 8sec video.
        Tr[0] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0])        
        T[0] = numpy.round(numpy.round(Tm[0], 4) +  numpy.round(Tr[0], 4), 4)
        # print (T[0][0])

        Tm[1] = frame_20000[1]
        Tr[1] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][1]) + (lat[dev1][1])        
        T[1] = numpy.round(numpy.round(Tm[1], 4) +  numpy.round(Tr[1], 4), 4)
        # print (T[1])

        Tm[2] = frame_20000[2]
        Tr[2] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][2]) + (lat[dev1][2])        
        T[2] = numpy.round(numpy.round(Tm[2], 4) +  numpy.round(Tr[2], 4), 4)
        # print (T[2])

        Tm[3] = frame_20000[3]
        Tr[3] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][3]) + (lat[dev1][3])        
        T[3] = numpy.round(numpy.round(Tm[3], 4) +  numpy.round(Tr[3], 4), 4)
        # print (T[3])

        Tm[4] = frame_20000[4]
        Tr[4] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][4]) + (lat[dev1][4])        
        T[4] = numpy.round(numpy.round(Tm[4], 4) +  numpy.round(Tr[4], 4), 4)
        # print (T[4])

        Tm[5] = frame_20000[5]
        Tr[5] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][5]) + (lat[dev1][5])        
        T[5] = numpy.round(numpy.round(Tm[5], 4) +  numpy.round(Tr[5], 4), 4)
        # print (T[5])

        Tm[6] = frame_20000[6]
        Tr[6] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][6]) + (lat[dev1][6])        
        T[6] = numpy.round(numpy.round(Tm[6], 4) +  numpy.round(Tr[6], 4), 4)
        # print (T[6])

        Tm[7] = frame_20000[7]
        Tr[7] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][7]) + (lat[dev1][7])        
        T[7] = numpy.round(numpy.round(Tm[7], 4) +  numpy.round(Tr[7], 4), 4)

        Tm[8] = frame_20000[8]
        Tr[8] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][8]) + (lat[dev1][8])        
        T[8] = numpy.round(numpy.round(Tm[8], 4) +  numpy.round(Tr[8], 4), 4)

        Tm[9] = frame_20000[9]
        Tr[9] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][9]) + (lat[dev1][9])        
        T[9] = numpy.round(numpy.round(Tm[9], 4) + numpy.round(Tr[9], 4), 4)

        Tm[10] = frame_20000[10]
        Tr[10] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][10]) + (lat[dev1][10])        
        T[10] = numpy.round(numpy.round(Tm[10], 4) +  numpy.round(Tr[10], 4), 4)
        # print (T[10])

    elif (task == "hightrain"):
        # print(hightrain[0]," ",type(rec))
        Tm[0] = hightrain[0]  # data size: 8sec video.
        Tr[0] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0])        
        T[0] = numpy.round(numpy.round(Tm[0], 4) +  numpy.round(Tr[0], 4), 4)
        # print (T[0][0])

        Tm[1] = hightrain[1]
        Tr[1] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][1]) + (lat[dev1][1])        
        T[1] = numpy.round(numpy.round(Tm[1], 4) +  numpy.round(Tr[1], 4), 4)
        # print (T[1])

        Tm[2] = hightrain[2]
        Tr[2] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][2]) + (lat[dev1][2])        
        T[2] = numpy.round(numpy.round(Tm[2], 4) +  numpy.round(Tr[2], 4), 4)
        # print (T[2])

        Tm[3] = hightrain[3]
        Tr[3] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][3]) + (lat[dev1][3])        
        T[3] = numpy.round(numpy.round(Tm[3], 4) +  numpy.round(Tr[3], 4), 4)
        # print (T[3])

        Tm[4] = hightrain[4]
        Tr[4] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][4]) + (lat[dev1][4])        
        T[4] = numpy.round(numpy.round(Tm[4], 4) +  numpy.round(Tr[4], 4), 4)
        # print (T[4])

        Tm[5] = hightrain[5]
        Tr[5] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][5]) + (lat[dev1][5])        
        T[5] = numpy.round(numpy.round(Tm[5], 4) +  numpy.round(Tr[5], 4), 4)
        # print (T[5])

        Tm[6] = hightrain[6]
        Tr[6] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][6]) + (lat[dev1][6])        
        T[6] = numpy.round(numpy.round(Tm[6], 4) +  numpy.round(Tr[6], 4), 4)
        # print (T[6])

        Tm[7] = hightrain[7]
        Tr[7] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][7]) + (lat[dev1][7])        
        T[7] = numpy.round(numpy.round(Tm[7], 4) +  numpy.round(Tr[8], 4), 4)

        Tm[8] = hightrain[8]
        Tr[8] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][8]) + (lat[dev1][8])        
        T[8] = numpy.round(numpy.round(Tm[8], 4) +  numpy.round(Tr[8], 4), 4)

        Tm[9] = hightrain[9]
        Tr[9] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][9]) + (lat[dev1][9])        
        T[9] = numpy.round(numpy.round(Tm[9], 4) + numpy.round(Tr[9], 4), 4)

        Tm[10] = hightrain[10]
        Tr[10] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][10]) + (lat[dev1][10])        
        T[10] = numpy.round(numpy.round(Tm[10], 4) +  numpy.round(Tr[10], 4), 4)
        # print (T[10])

    elif (task == "lowtrain"):
        # print(lowtrain[0]," ",type(rec))
        Tm[0] = lowtrain[0]  # data size: 8sec video.
        Tr[0] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0])        
        T[0] = numpy.round(numpy.round(Tm[0], 4) +  numpy.round(Tr[0], 4), 4)
        # print (T[0][0])

        Tm[1] = lowtrain[1]
        Tr[1] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][1]) + (lat[dev1][1])        
        T[1] = numpy.round(numpy.round(Tm[1], 4) +  numpy.round(Tr[1], 4), 4)
        # print (T[1])

        Tm[2] = lowtrain[2]
        Tr[2] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][2]) + (lat[dev1][2])        
        T[2] = numpy.round(numpy.round(Tm[2], 4) +  numpy.round(Tr[2], 4), 4)
        # print (T[2])

        Tm[3] = lowtrain[3]
        Tr[3] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][3]) + (lat[dev1][3])        
        T[3] = numpy.round(numpy.round(Tm[3], 4) +  numpy.round(Tr[3], 4), 4)
        # print (T[3])

        Tm[4] = lowtrain[4]
        Tr[4] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][4]) + (lat[dev1][4])        
        T[4] = numpy.round(numpy.round(Tm[4], 4) +  numpy.round(Tr[4], 4), 4)
        # print (T[4])

        Tm[5] = lowtrain[5]
        Tr[5] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][5]) + (lat[dev1][5])        
        T[5] = numpy.round(numpy.round(Tm[5], 4) +  numpy.round(Tr[5], 4), 4)
        # print (T[5])

        Tm[6] = lowtrain[6]
        Tr[6] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][6]) + (lat[dev1][6])        
        T[6] = numpy.round(numpy.round(Tm[6], 4) +  numpy.round(Tr[6], 4), 4)
        # print (T[6])

        Tm[7] = lowtrain[7]
        Tr[7] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][7]) + (lat[dev1][7])        
        T[7] = numpy.round(numpy.round(Tm[7], 4) +  numpy.round(Tr[7], 4), 4)

        Tm[8] = lowtrain[8]
        Tr[8] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][8]) + (lat[dev1][8])        
        T[8] = numpy.round(numpy.round(Tm[8], 4) +  numpy.round(Tr[8], 4), 4)

        Tm[9] = lowtrain[9]
        Tr[9] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][9]) + (lat[dev1][9])        
        T[9] = numpy.round(numpy.round(Tm[9], 4) + numpy.round(Tr[9], 4), 4)

        Tm[10] = lowtrain[10]
        Tr[10] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][10]) + (lat[dev1][10])        
        T[10] = numpy.round(numpy.round(Tm[10], 4) +  numpy.round(Tr[10], 4), 4)
        # print (T[10])

    elif (task == "inference"):
        # print(inference[0]," ",type(rec))
        Tm[0] = inference[0]  # data size: 8sec video.
        Tr[0] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][0]) + (lat[dev2][0]))  
        T[0] = numpy.round(numpy.round(Tm[0], 4) +  numpy.round(Tr[0], 4), 4)
        # print (T[0][0])

        Tm[1] = inference[1]
        Tr[1] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][1]) + (lat[dev1][1]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][1]) + (lat[dev2][1]))          
        T[1] = numpy.round(numpy.round(Tm[1], 4) +  numpy.round(Tr[1], 4), 4)
        # print (T[1])

        Tm[2] = inference[2]
        Tr[2] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][2]) + (lat[dev1][2]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][2]) + (lat[dev2][2]))         
        T[2] = numpy.round(numpy.round(Tm[2], 4) +  numpy.round(Tr[2], 4), 4)
        # print (T[2])

        Tm[3] = inference[3]
        Tr[3] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][3]) + (lat[dev1][3]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][3]) + (lat[dev2][3]))          
        T[3] = numpy.round(numpy.round(Tm[3], 4) +  numpy.round(Tr[3], 4), 4)
        # print (T[3])

        Tm[4] = inference[4]
        Tr[4] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][4]) + (lat[dev1][4]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][4]) + (lat[dev2][4]))         
        T[4] = numpy.round(numpy.round(Tm[4], 4) +  numpy.round(Tr[4], 4), 4)
        # print (T[4])

        Tm[5] = inference[5]
        Tr[5] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][5]) + (lat[dev1][5]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][5]) + (lat[dev2][5]))
        T[5] = numpy.round(numpy.round(Tm[5], 4) +  numpy.round(Tr[5], 4), 4)
        # print (T[5])

        Tm[6] = inference[6]
        Tr[6] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][6]) + (lat[dev1][6]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][6]) + (lat[dev2][6])) 
        T[6] = numpy.round(numpy.round(Tm[6], 4) +  numpy.round(Tr[6], 4), 4)
        # print (T[6])

        Tm[7] = inference[7]
        Tr[7] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][7]) + (lat[dev1][7]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][7]) + (lat[dev2][7]))  
        T[7] = numpy.round(numpy.round(Tm[7], 4) +  numpy.round(Tr[7], 4), 4)
        # print (T[7])


        Tm[8] = inference[8]
        Tr[8] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][8]) + (lat[dev1][8]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][8]) + (lat[dev2][8]))
        T[8] = numpy.round(numpy.round(Tm[8], 4) +  numpy.round(Tr[8], 4), 4)
        # print (T[8])

        Tm[9] = inference[9]
        Tr[9] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][9]) + (lat[dev1][9]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][9]) + (lat[dev2][9]))
        T[9] = numpy.round(numpy.round(Tm[9], 4) + numpy.round(Tr[9], 4), 4)
        # print (T[9])

        Tm[10] = inference[10]
        Tr[10] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][10]) + (lat[dev1][10]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][10]) + (lat[dev2][10]))         
        T[10] = numpy.round(numpy.round(Tm[10], 4) +  numpy.round(Tr[10], 4), 4)
        # print (T[10])

    elif (task == "transcode"):
        ###print(encode_20000[0]," ",type(rec))
        Tm[0] = encode_20000[0]  # data size: 8sec video.
        Tr[0] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][0]) + (lat[dev2][0]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][0]) + (lat[dev3][0]))
        T[0] = numpy.round(numpy.round(Tm[0], 4) +  numpy.round(Tr[0], 4), 4)
        #print (T[0])

        Tm[1] = encode_20000[1]
        Tr[1] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][1]) + (lat[dev1][1]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][1]) + (lat[dev2][1]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][1]) + (lat[dev3][1]))        
        T[1] = numpy.round(numpy.round(Tm[1], 4) +  numpy.round(Tr[1], 4), 4)
        #print (T[1])

        Tm[2] = encode_20000[2]
        Tr[2] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][2]) + (lat[dev1][2]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][2]) + (lat[dev2][2]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][2]) + (lat[dev3][2]))        
        T[2] = numpy.round(numpy.round(Tm[2], 4) +  numpy.round(Tr[2], 4), 4)
        #print (T[2])

        Tm[3] = encode_20000[3]
        Tr[3] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][3]) + (lat[dev1][3]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][3]) + (lat[dev2][3]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][3]) + (lat[dev3][3]))        
        T[3] = numpy.round(numpy.round(Tm[3], 4) +  numpy.round(Tr[3], 4), 4)
        #print (T[3])

        Tm[4] = encode_20000[4]
        Tr[4] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][4]) + (lat[dev1][4]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][4]) + (lat[dev2][4]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][4]) + (lat[dev3][4]))        
        T[4] = numpy.round(numpy.round(Tm[4], 4) +  numpy.round(Tr[4], 4), 4)
        #print (T[4])

        Tm[5] = encode_20000[5]
        Tr[5] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][2]) + (lat[dev1][5]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][5]) + (lat[dev2][5]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][5]) + (lat[dev3][5]))        
        T[5] = numpy.round(numpy.round(Tm[5], 4) +  numpy.round(Tr[5], 4), 4)
        #print (T[5])

        Tm[6] = encode_20000[6]
        Tr[6] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][6]) + (lat[dev1][6]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][6]) + (lat[dev2][6]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][6]) + (lat[dev3][6]))        
        T[6] = numpy.round(numpy.round(Tm[6], 4) +  numpy.round(Tr[6], 4), 4)
        #print (T[6])

        Tm[7] = encode_20000[7]
        Tr[7] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][7]) + (lat[dev1][7]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][7]) + (lat[dev2][7]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][7]) + (lat[dev3][7]))        
        T[7] = numpy.round(numpy.round(Tm[7], 4) +  numpy.round(Tr[7], 4), 4)

        Tm[8] = encode_20000[8]
        Tr[8] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][8]) + (lat[dev1][8]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][8]) + (lat[dev2][8]),
                ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][8]) + (lat[dev3][8]))        
        T[8] = numpy.round(numpy.round(Tm[8], 4) +  numpy.round(Tr[8], 4), 4)

        Tm[9] = encode_20000[9]
        Tr[9] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][9]) + (lat[dev1][9])
        T[9] = numpy.round(numpy.round(Tm[9], 4) + numpy.round(Tr[9], 4), 4)

        Tm[10] = encode_20000[10]
        Tr[10] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][10]) + (lat[dev1][10])
        T[10] = numpy.round(numpy.round(Tm[10], 4) + numpy.round(Tr[10], 4), 4)

    elif (task == "package"):
        ###print(encode_20000[0]," ",type(rec))
        Tm[0] = encode_20000[0]  # data size: 8sec video.
        Tr[0] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0])
        T[0] = numpy.round(numpy.round(Tm[0], 4) +  numpy.round(Tr[0], 4), 4)
        #print (T[0])

        Tm[1] = encode_20000[1]
        Tr[1] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][1]) + (lat[dev1][1])        
        T[1] = numpy.round(numpy.round(Tm[1], 4) +  numpy.round(Tr[1], 4), 4)
        #print (T[1])

        Tm[2] = encode_20000[2]
        Tr[2] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][2]) + (lat[dev1][2])        
        T[2] = numpy.round(numpy.round(Tm[2], 4) +  numpy.round(Tr[2], 4), 4)
        #print (T[2])

        Tm[3] = encode_20000[3]
        Tr[3] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][3]) + (lat[dev1][3])        
        T[3] = numpy.round(numpy.round(Tm[3], 4) +  numpy.round(Tr[3], 4), 4)
        #print (T[3])

        Tm[4] = encode_20000[4]
        Tr[4] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][4]) + (lat[dev1][4])        
        T[4] = numpy.round(numpy.round(Tm[4], 4) +  numpy.round(Tr[4], 4), 4)
        #print (T[4])

        Tm[5] = encode_20000[5]
        Tr[5] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][5]) + (lat[dev1][5])        
        T[5] = numpy.round(numpy.round(Tm[5], 4) +  numpy.round(Tr[5], 4), 4)
        #print (T[5])

        Tm[6] = encode_20000[6]
        Tr[6] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][6]) + (lat[dev1][6])        
        T[6] = numpy.round(numpy.round(Tm[6], 4) +  numpy.round(Tr[6], 4), 4)
        #print (T[6])

        Tm[7] = encode_20000[7]
        Tr[7] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][7]) + (lat[dev1][7])        
        T[7] = numpy.round(numpy.round(Tm[7], 4) +  numpy.round(Tr[7], 4), 4)

        Tm[8] = encode_20000[8]
        Tr[8] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][8]) + (lat[dev1][8])        
        T[8] = numpy.round(numpy.round(Tm[8], 4) +  numpy.round(Tr[8], 4), 4)


        Tm[9] = encode_20000[9]
        Tr[9] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][9]) + (lat[dev1][9])        
        T[9] = numpy.round(numpy.round(Tm[9], 4) + numpy.round(Tr[9], 4), 4)


        Tm[10] = encode_20000[10]
        Tr[10] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][10]) + (lat[dev1][10])
        T[10] = numpy.round(numpy.round(Tm[10], 4) +  numpy.round(Tr[10], 4), 4)
        # print (T[10])

    return (Tm, Tr, T)