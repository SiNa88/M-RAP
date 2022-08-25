import numpy
import subprocess
import os
import json
import time
import sys
import random

#Device_ifo=[{'id': 0, 'name':"Wein_Medium",'IPT': 7200, 'cpu': {'core_num': 2, 'cash': 4, 'speed': 7200},'network': {'up_Bw': 220, 'down_Bw': 220, 'Latency':7}, 'storage': 10, 'RAM': 4},
#            {'id': 1, 'name':"Wein_Small",'IPT': 7100, 'cpu': {'core_num': 2, 'cash': 4, 'speed': 7100}, 'network': {'up_Bw': 220, 'down_Bw': 220, 'Latency':7}, 'storage': 10,'RAM': 2},
#            {'id': 2, 'name':"Wein_Tiny",'IPT': 3700, 'cpu': {'core_num': 1, 'cash': 4, 'speed': 3700}, 'network': {'up_Bw': 220, 'down_Bw': 220, 'Latency':7}, 'storage': 10,'RAM': 1},
#            {'id': 3, 'name':"ZHR_Medium", 'IPT': 7200, 'cpu': {'core_num': 2, 'cash': 4, 'speed': 7200},'network': {'up_Bw': 80, 'down_Bw': 80, 'Latency':24}, 'storage': 10, 'RAM': 4},
#            {'id': 4, 'name':"MUC_Medium",'IPT': 7200, 'cpu': {'core_num': 2, 'cash': 4, 'speed': 7200},'network': {'up_Bw': 140, 'down_Bw': 140, 'Latency':13}, 'storage': 10, 'RAM': 4},
#            {'id': 5,'name':"edge", 'IPT': 12000, 'cpu': {'core_num': 4, 'cash': 4, 'speed': 12000}, 'network': {'up_Bw': 220, 'down_Bw': 220, 'Latency':12}, 'storage': 10,'RAM': 8},
#           {'id': 6,'name':"AAU-Large", 'IPT': 58000, 'cpu': {'core_num': 12, 'cash': 4, 'speed': 58000}, 'network': {'up_Bw': 940, 'down_Bw': 940, 'Latency':2}, 'storage': 32,'RAM': 32},
#            {'id': 7,'name':"AAU-Medium", 'IPT': 21700, 'cpu': {'core_num': 8, 'cash': 4, 'speed': 21700}, 'network': {'up_Bw': 920, 'down_Bw': 920, 'Latency':2}, 'storage': 32,'RAM': 16},
#            {'id': 8,'name':"Jetson", 'IPT': 4080, 'cpu': {'core_num': 4, 'cash': 4, 'speed': 4080}, 'network': {'up_Bw': 840, 'down_Bw': 840, 'Latency':2}, 'storage': 64,'RAM': 4},
#            {'id': 9,'name':"RPi4", 'IPT': 5100, 'cpu': {'core_num': 4, 'cash': 4, 'speed': 5100}, 'network': {'up_Bw': 800, 'down_Bw': 800, 'Latency':2}, 'storage': 64,'RAM': 4}]

# addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
# del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]

def comp_times_DeltaInterval_0_video(task, dev_new, dev1, dev2, dev3):
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


        # seg_size = 80000 #(10KB)
        # video_size = 8000000000 #(1GB)
        index_of_segment = 4
        seg_size = [286720, 2457600, 3440640, 14400000, 20971520]  # bits
        video_size = [2000000, 14000000, 28000000, 60000000, 204800000]  # bits

        # Converting string to list
        tasks0 = ["encode_20000", "frame_20000", "lowtrain", "hightrain", "inference", "transcode", "package"]  # sys.argv[1].strip("][").split(",")
        # print(type(tasks0))

        num_of_apps = 1
        newtasks = [0 for i in range(len(tasks0) * num_of_apps)]
        for k in range(num_of_apps):
            for i in range(len(tasks0)):
                newtasks[i + (k * len(tasks0))] = tasks0[i] + str(k)

        resources = ["vm-exo-med","vm-exo-SMALL", "vm-exo-Tiny","ZHR-exo-med","MUC-exo-med","Edge-kla", "AAu(large)","lenovo", "jetson", "rpi4", "SOF-exo-HUge"]

        # 7 & 7 & 7 &  7 & 26 & 26 & 24 & 24 & 24 & 18 & 18 & 13 & 13 &  &  1 & 1  & 1 & 1 & 1
        # print (len(lat))
        # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
        # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]

        lat = [[ 0.5,0.5,0.5,12.5,7.3,4.8,7.2,7.2,7.5,7.5,15],
                    [0.5,0.5,0.5,12.5,7.3,4.8,7.2,7.2,7.5,7.5,15],
                    [0.5, 0.5,0.5,12.5,7.3,4.8,7.2,7.2,7.5,7.5,15],
                    [12.5,12.5,12.5,0.5,6.7,16.6,23.2,23.6,23.2,23.6,25.9],
                    [7.3,7.3,7.3,6.7,0.5,11.5,12.2,12.5,12.6,12.6,21],
                    [4.8,4.8,4.8,16.6,11.5,0.5,11.4,11.5,12,11.5,10],
                    [7.2,7.2,7.2,23.2,12.2,11.4,0.5,0.5,0.5,0.5,22.4],
                    [7.2,7.2,7.2,23.6,12.5,11.5,0.5,0.5,0.5,0.5,22.8],
                    [7.5,7.5,7.5,23.2,12.6,12,0.5,0.5,0.5,0.5, 22.8],
                    [7.5, 7.5, 7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5,22.6],
                    [15,15,15,25.9,21,10,22.4,22.8, 22.8,22.6,0.5]]

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
        index_of_segment = 4
        seg_size = [286720, 2457600, 3440640, 14400000, 20971520]  # bits
        video_size = [2000000, 14000000, 28000000, 60000000, 204800000]  # bits
        
        T = [0 for i in range(len(tasks0))]
        Tm = [0 for i in range(len(tasks0))]
        Tr = [0 for i in range(len(tasks0))]

        if (task == "encode_20000"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "frame_20000"):
            ###print(frame_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = frame_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "hightrain"):
            # print(hightrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = hightrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "lowtrain"):
            # print(lowtrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = lowtrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "inference"):
            # print(inference[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = inference[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]))  
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "transcode"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "package"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        return (Tm, Tr, T)
###########################################################################################################################################
# time interval 1
###########################################################################################################################################
    # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
    # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
    #             V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)

def comp_times_DeltaInterval_1_video(task, dev_new, dev1, dev2, dev3):

        #      V-Exo(small) V-Exo(tiny)  Z-Exo(med) M-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4 NJN   RPi4  SOF-exo-HUge
        encode_200 = [0.44, 0.46, 0.4, 0.4, 0.28, 0.17, 0.33, 1.9, 2.16, 1.9, 2.16, 0.23]  # seconds
        encode_1500 = [0.65, 1.14, 0.62, 0.62, 0.5, 0.36, 0.42, 2.63, 3.19,2.63, 3.19, 0.43]  # seconds
        encode_3000 = [0.9, 1.6, 0.89, 0.89, 0.63, 0.47, 0.59, 3.48, 4.4,3.48, 4.4, 0.55]  # seconds
        encode_6500 = [2.7, 4.9, 2.58, 2.58, 1.45, 1.22, 1.59, 9.68, 11.8,9.68, 11.8, 1.3]  # seconds
        encode_20000 = [6.2, 11, 6.1, 6.1, 3.1, 2.7, 3.16, 20.64, 28,20.64, 28, 2.6]  # seconds

        #           V-Exo(small) V-Exo(tiny)  Z-Exo(med) M-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   SOF-exo-HUge
        frame_200 = [ 0.44, 0.5, 0.8, 0.8, 0.5, 0.39, 0.35, 2, 4, 2, 4,0.77]
        frame_1500 = [1.8, 2, 1.8, 2, 2.46, 2.1, 9.4, 11,9.4, 11, 1.83]
        frame_3000 = [2.6, 2.8, 2.5, 2.5, 2.9, 3.8, 3.2, 14, 14,14, 14, 2]
        frame_6500 = [9.4, 9.8, 8.7, 8.7, 9.9, 14.2, 11.8, 55, 49, 55, 49,8.4]
        frame_20000 = [18.2, 21, 18.5, 18.5, 20.9, 31, 26, 117, 112,117, 112, 17.5]

        # 		   V-Exo(small) V-Exo(tiny)  Z-Exo(med) M-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   SOF-exo-HUge
        inference = [0.29, 0.3, 0.26, 0.28, 0.225, 0.282, 0.28, 1.94, 1.1,1.94, 1.1, 0.246]
        lowtrain = [32, 35, 26.5, 26.5, 23, 17, 18, 152, 102, 152, 102,19.6]  # seconds
        hightrain = [109, 115, 103, 103, 77, 33, 57, 232, 467,232, 467, 62.5]  # seconds

        # Converting string to list
        tasks0 = ["encode_20000", "frame_20000", "lowtrain", "hightrain", "inference", "transcode", "package"]  # sys.argv[1].strip("][").split(",")
        # print(type(tasks0))

        num_of_apps = 1
        newtasks = [0 for i in range(len(tasks0) * num_of_apps)]
        for k in range(num_of_apps):
            for i in range(len(tasks0)):
                newtasks[i + (k * len(tasks0))] = tasks0[i] + str(k)

        resources = ["vm-exo-SMALL", "vm-exo-Tiny", "ZHR-exo-med", "MUC-exo-med", "Edge-kla", "AAu(large)", "lenovo",
                     "jetson", "rpi4",  "jetson", "rpi4","SOF-exo-HUge"]
        lat = [[0.5, 0.5, 12.5, 7.3, 4.8, 7.2, 7.2, 7.5, 7.5,7.5,7.5, 15],
            [0.5, 0.5, 12.5, 7.3, 4.8, 7.2, 7.2, 7.5, 7.5,7.5,7.5, 15],
            [12.5, 12.5, 0.5, 6.7, 16.6, 23.2, 23.6, 23.2, 23.6, 23.2, 23.2, 25.9],
            [7.3, 7.3, 6.7, 0.5, 11.5, 12.2, 12.5, 12.6, 12.6, 12.6, 12.6, 21],
            [4.8, 4.8, 16.6, 11.5, 0.5, 11.4, 11.5, 12, 11.5, 12,12, 10],
            [7.2, 7.2, 23.2, 12.2, 11.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 22.4],
            [7.2, 7.2, 23.6, 12.5, 11.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 22.8],
            [7.5, 7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 22.8],
            [7.5, 7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 22.6],
            [7.5, 7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 22.8],
            [7.5, 7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 22.6],
            [15, 15, 25.9, 21, 10, 22.4, 22.8, 22.8, 22.6,22.8, 22.6, 0.5]]

        SIZE = 208
        BW_r = [[13000, 13000, 1950, 3000, 5000, 950, 900, 900, 900,900, 900, 1500],
             [13000, 13000, 1950, 3000, 5000, 950, 900, 900, 900, 900, 900, 1500],
             [1950, 1950, 1200, 3200, 1400, 900, 850, 700, 770, 700, 770, 900],
             [3000, 3000, 3200, 1200, 2100, 930, 900, 900, 850,900, 850, 1100],
             [5000, 5000, 1400, 2100, 12000, 930, 860, 840, 850, 840, 850, 1200],
             [950, 950, 900, 930, 930, 930, 930, 930, 850,930, 850, 920],
             [900, 900, 850, 900, 860, 930, 860, 920, 850,920, 850, 900],
             [900, 900, 700, 900, 840, 930, 850, 920, 920,920, 920, 900],
             [900, 900, 770, 850, 850, 850, 850, 920, 850, 920, 850, 850],
             [900, 900, 700, 900, 840, 930, 850, 920, 920,920, 920, 900],
             [900, 900, 770, 850, 850, 850, 850, 920, 850,920, 850, 850],
             [500, 1500, 900, 1100, 1200, 920, 900, 900, 850, 900, 850, 12000]]
        index_of_segment = 4
        seg_size = [286720, 2457600, 3440640, 14400000, 20971520]  # bits
        video_size = [2000000, 14000000, 28000000, 60000000, 204800000]  # bits
        
        T = [0 for i in range(len(tasks0))]
        Tm = [0 for i in range(len(tasks0))]
        Tr = [0 for i in range(len(tasks0))]

        if (task == "encode_20000"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "frame_20000"):
            ###print(frame_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = frame_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "hightrain"):
            # print(hightrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = hightrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "lowtrain"):
            # print(lowtrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = lowtrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "inference"):
            # print(inference[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = inference[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]))  
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "transcode"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "package"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        return (Tm, Tr, T)
    ###########################################################################################################################################
    # time interval 2
    ###########################################################################################################################################
    # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
    # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
    #             V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)

def comp_times_DeltaInterval_2_video(task, dev_new, dev1, dev2, dev3):

        #      V-Exo(small)  Z-Exo(med) M-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny) SOF-exo-HUge
        encode_200 = [0.44, 0.4, 0.4, 0.28, 0.17, 0.33, 1.9, 2.16, 1.9, 2.16,0.44, 0.46, 0.23]  # seconds
        encode_1500 = [0.65, 0.62, 0.62, 0.5, 0.36, 0.42, 2.63, 3.19, 2.63, 3.19,0.65, 1.14, 0.43]  # seconds
        encode_3000 = [0.9,0.89, 0.89, 0.63, 0.47, 0.59, 3.48, 4.4, 3.48, 4.4,0.9, 1.6, 0.55]  # seconds
        encode_6500 = [2.7, 2.58, 2.58, 1.45, 1.22, 1.59, 9.68, 11.8, 9.68, 11.8,2.7, 4.9, 1.3]  # seconds
        encode_20000 = [6.2, 6.1, 6.1, 3.1, 2.7, 3.16, 20.64, 28, 20.64, 28,6.2, 11, 2.6]  # seconds

        #           V-Exo(small)  Z-Exo(med) M-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny)  SOF-exo-HUge
        frame_200 = [0.44, 0.8, 0.8, 0.5, 0.39, 0.35, 2, 4, 2, 4,0.44, 0.5, 0.77]
        frame_1500 = [1.8, 1.8, 2, 2.46, 2.1, 9.4, 11, 9.4, 11,1.8, 2, 1.83]
        frame_3000 = [2.6, 2.5, 2.5, 2.9, 3.8, 3.2, 14, 14, 14, 14,2.6, 2.8, 2]
        frame_6500 = [9.4,8.7, 8.7, 9.9, 14.2, 11.8, 55, 49, 55, 49,9.4, 9.8, 8.4]
        frame_20000 = [18.2, 18.5, 18.5, 20.9, 31, 26, 117, 112, 117, 112,18.2, 21, 17.5]

        # 		   V-Exo(small) Z-Exo(med) M-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny)  SOF-exo-HUge
        inference = [0.29, 0.26, 0.28, 0.225, 0.282, 0.28, 1.94, 1.1, 1.94, 1.1,0.29, 0.3, 0.246]
        lowtrain = [32,26.5, 26.5, 23, 17, 18, 152, 102, 152, 102, 32, 35,19.6]  # seconds
        hightrain = [109, 103, 103, 77, 33, 57, 232, 467, 232, 467,109, 115, 62.5]  # seconds

        # Converting string to list
        tasks0 = ["encode_20000", "frame_20000", "lowtrain", "hightrain", "inference", "transcode", "package"]  # sys.argv[1].strip("][").split(",")
        # print(type(tasks0))

        resources = ["vm-exo-SMALL", "ZHR-exo-med", "MUC-exo-med", "Edge-kla", "AAu(large)", "lenovo",
                     "jetson", "rpi4", "jetson", "rpi4","vm-exo-SMALL", "vm-exo-Tiny", "SOF-exo-HUge"]


        lat = [[0.5, 12.5, 4.8, 7.2, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 15],
            [12.5, 0.5, 16.6, 23.2, 23.6, 23.2, 23.6, 23.2, 23.2, 12.5, 12.5, 25.9],
            [7.3, 6.7, 0.5, 11.5, 12.2, 12.5, 12.6, 12.6, 12.6, 12.6, 7.3, 7.3, 21],
            [4.8, 16.6, 0.5, 11.4, 11.5, 12, 11.5, 12, 12, 4.8, 4.8, 10],
            [7.2, 23.2, 11.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.2, 7.2, 22.4],
            [7.2, 23.6, 11.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.2, 7.2, 22.8],
            [7.5, 23.2, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 22.8],
            [7.5, 23.2, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 22.6],
            [7.5, 23.2, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 22.8],
            [7.5, 23.2, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5,7.5, 22.6],
            [0.5, 12.5, 4.8, 7.2, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 15],
            [0.5, 12.5, 4.8, 7.2, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 15],
            [15, 25.9, 10, 22.4, 22.8, 22.8, 22.6, 22.8, 22.6, 15, 15, 0.5]]

        SIZE = 208
        BW_r = [[13000, 1950, 3000, 5000, 950, 900, 900, 900, 900, 900, 13000, 13000, 1500],
             [1950, 1200, 3200, 1400, 900, 850, 700, 770, 700, 770, 1950, 1950, 900],
             [3000, 3200, 1200, 2100, 930, 900, 900, 850, 900, 850, 3000, 3000, 1100],
             [5000, 1400, 2100, 12000, 930, 860, 840, 850, 840, 850, 5000, 5000, 1200],
             [950, 900, 930, 930, 930, 930, 930, 850, 930, 850, 950, 950, 920],
             [900, 850, 900, 860, 930, 860, 920, 850, 920, 850, 900, 900, 900],
             [900, 700, 900, 840, 930, 850, 920, 920, 920, 920, 900, 900, 900],
             [900, 770, 850, 850, 850, 850, 920, 850, 920, 850, 900, 900, 850],
             [900, 700, 900, 840, 930, 850, 920, 920, 920, 920, 900, 900, 900],
             [900, 770, 850, 850, 850, 850, 920, 850, 920, 850, 900, 900, 850],
             [13000, 1950, 3000, 5000, 950, 900, 900, 900, 900, 900, 900, 900, 13000, 13000, 1500],
             [13000, 1950, 3000, 5000, 950, 900, 900, 900, 900, 900, 13000, 13000, 1500],
             [500, 900, 1100, 1200, 920, 900, 900, 850, 900, 850, 1500, 1500, 12000]]

        index_of_segment = 4
        seg_size = [286720, 2457600, 3440640, 14400000, 20971520]  # bits
        video_size = [2000000, 14000000, 28000000, 60000000, 204800000]  # bits
        
        T = [0 for i in range(len(tasks0))]
        Tm = [0 for i in range(len(tasks0))]
        Tr = [0 for i in range(len(tasks0))]

        if (task == "encode_20000"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "frame_20000"):
            ###print(frame_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = frame_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "hightrain"):
            # print(hightrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = hightrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "lowtrain"):
            # print(lowtrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = lowtrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "inference"):
            # print(inference[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = inference[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]))  
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "transcode"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "package"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        return (Tm, Tr, T)


###########################################################################################################################################
# time interval 3
###########################################################################################################################################
# addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
# del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
#             V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)

def comp_times_DeltaInterval_3_video(task, dev_new, dev1, dev2, dev3):
        #      V-Exo(small)  Z-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 SOF-exo-HUge
        encode_200 = [0.44, 0.4, 0.28, 0.17, 0.33, 1.9, 2.16, 1.9, 2.16, 0.44, 0.46,1.9, 2.16, 0.23]  # seconds
        encode_1500 = [0.65, 0.62, 0.5, 0.36, 0.42, 2.63, 3.19, 2.63, 3.19, 0.65, 1.14,2.63, 3.19, 0.43]  # seconds
        encode_3000 = [0.9, 0.89,0.63, 0.47, 0.59, 3.48, 4.4, 3.48, 4.4, 0.9, 1.6, 3.48, 4.4, 0.55]  # seconds
        encode_6500 = [2.7, 2.58, 1.45, 1.22, 1.59, 9.68, 11.8, 9.68, 11.8, 2.7, 4.9, 9.68, 11.8, 1.3]  # seconds
        encode_20000 = [6.2, 6.1, 3.1, 2.7, 3.16, 20.64, 28, 20.64, 28, 6.2, 11,20.64, 28, 2.6]  # seconds

        #           V-Exo(small)  Z-Exo(med)  Edge-kla AAu(large) Lenovo  NJN   RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 SOF-exo-HUge
        frame_200 = [0.44, 0.8, 0.5, 0.39, 0.35, 2, 4, 2, 4, 0.44, 0.5, 2, 4, 0.77]
        frame_1500 = [1.8, 1.8, 2.46, 2.1, 9.4, 11, 9.4, 11, 1.8, 2, 9.4, 11, 1.83]
        frame_3000 = [2.6, 2.5, 2.9, 3.8, 3.2, 14, 14, 14, 14, 2.6, 2.8,14, 14, 2]
        frame_6500 = [9.4, 8.7,9.9, 14.2, 11.8, 55, 49, 55, 49, 9.4, 9.8,55, 49,  8.4]
        frame_20000 = [18.2, 18.5, 20.9, 31, 26, 117, 112, 117, 112, 18.2, 21,  117, 112,17.5]

        # 		   V-Exo(small) Z-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 SOF-exo-HUge
        inference = [0.29, 0.26, 0.225, 0.282, 0.28, 1.94, 1.1, 1.94, 1.1, 0.29, 0.3,1.94, 1.1, 0.246]
        lowtrain = [32, 26.5, 23, 17, 18, 152, 102, 152, 102, 32, 35, 152, 102,19.6]  # seconds
        hightrain = [109, 103, 77, 33, 57, 232, 467, 232, 467, 109, 115, 232, 467, 62.5]  # seconds

        # Converting string to list
        tasks0 = ["encode_20000", "frame_20000", "lowtrain", "hightrain", "inference", "transcode", "package"]  # sys.argv[1].strip("][").split(",")
        # print(type(tasks0))

        resources = ["vm-exo-SMALL", "ZHR-exo-med", "Edge-kla", "AAu(large)", "lenovo",
                    "jetson", "rpi4", "jetson", "rpi4", "vm-exo-SMALL", "vm-exo-Tiny", "jetson", "rpi4","SOF-exo-HUge"]


        lat = [[0.5, 12.5, 7.3, 4.8, 7.2, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5, 15],
                [12.5, 0.5, 6.7, 16.6, 23.2, 23.6, 23.2, 23.6, 23.2, 23.2, 12.5, 12.5, 23.2, 23.2, 25.9],
                [7.3, 6.7, 0.5, 11.5, 12.2, 12.5, 12.6, 12.6, 12.6, 12.6, 7.3, 7.3, 23.2, 23.2, 21],
                [4.8, 16.6, 11.5, 0.5, 11.4, 11.5, 12, 11.5, 12, 12, 4.8, 4.8, 12.6, 12.6, 12, 12, 10],
                [7.2, 23.2, 12.2, 11.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.2, 7.2, 0.5, 0.5, 12, 12, 22.4],
                [7.2, 23.6, 12.5, 11.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.2, 7.2, 0.5, 0.5, 22.8],
                [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5, 22.8],
                [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5, 22.6],
                [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5, 22.8],
                [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5, 22.6],
                [0.5, 12.5, 7.3, 4.8, 7.2, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 0.5, 0.5, 15],
                [0.5, 12.5, 7.3, 4.8, 7.2, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 0.5, 0.5, 15],
                [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5, 22.8],
                [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5, 22.6],
                [15, 25.9, 21, 10, 22.4, 22.8, 22.8, 22.8, 22.6, 22.6, 15, 15, 22.8, 22.6, 0.5]]

        SIZE = 208
        BW_r = [[13000, 1950, 5000, 950, 900, 900, 900, 900, 900, 13000, 13000, 900, 900, 1500],
                [1950, 1200, 1400, 900, 850, 700, 770, 700, 770, 1950, 1950, 700, 770, 900],
                [5000, 1400, 12000, 930, 860, 840, 850, 840, 850, 5000, 5000, 840, 850, 1200],
                [950, 900, 930, 930, 930, 930, 850, 930, 850, 950, 950, 930, 850, 920],
                [900, 850, 860, 930, 860, 920, 850, 920, 850, 900, 900, 850, 850, 900],
                [900, 700, 840, 930, 850, 920, 920, 920, 920, 900, 900, 920, 920, 900],
                [900, 770, 850, 850, 850, 920, 850, 920, 850, 900, 900, 920, 920, 920, 850, 850],
                [900, 700, 840, 930, 850, 920, 920, 920, 920, 900, 900, 920, 920, 900],
                [900, 770, 850, 850, 850, 920, 850, 920, 850, 900, 900, 920, 850, 850],
                [13000, 1950, 5000, 950, 900, 900, 900, 900, 900, 13000, 13000, 900, 900, 1500],
                [13000, 1950, 5000, 950, 900, 900, 900, 900, 900, 13000, 13000, 900, 900, 1500],
                [900, 700, 840, 930, 850, 920, 920, 920, 920, 900, 900, 920, 900, 900],
                [900, 770, 850, 850, 850, 920, 850, 920, 850, 900, 900, 900, 920, 850],
                [500, 900, 1200, 920, 900, 900, 850, 900, 850, 1500, 1500, 900, 850, 12000]]

        index_of_segment = 4
        seg_size = [286720, 2457600, 3440640, 14400000, 20971520]  # bits
        video_size = [2000000, 14000000, 28000000, 60000000, 204800000]  # bits
        
        T = [0 for i in range(len(tasks0))]
        Tm = [0 for i in range(len(tasks0))]
        Tr = [0 for i in range(len(tasks0))]

        if (task == "encode_20000"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "frame_20000"):
            ###print(frame_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = frame_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "hightrain"):
            # print(hightrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = hightrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "lowtrain"):
            # print(lowtrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = lowtrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "inference"):
            # print(inference[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = inference[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]))  
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "transcode"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "package"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        return (Tm, Tr, T)


    ###########################################################################################################################################
    # time interval 4
    ###########################################################################################################################################
    # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
    # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
    #             V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)

def comp_times_DeltaInterval_4_video(task, dev_new, dev1, dev2, dev3):
        #      V-Exo(small)  Z-Exo(med) Edge-kla AAu(large)  NJN   RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 V-Exo(med) V-Exo(small) SOF-exo-HUge
        encode_200 = [0.44, 0.4, 0.28, 0.17, 1.9, 2.16, 1.9, 2.16, 0.44, 0.46, 1.9, 2.16,0.4, 0.44,0.23]  # seconds
        encode_1500 = [0.65, 0.62, 0.5, 0.36, 2.63, 3.19, 2.63, 3.19, 0.65, 1.14, 2.63, 3.19, 0.62,0.65,0.43]  # seconds
        encode_3000 = [0.9, 0.89, 0.63, 0.47, 3.48, 4.4, 3.48, 4.4, 0.9, 1.6, 3.48, 4.4, 0.89,0.9, 0.55]  # seconds
        encode_6500 = [2.7, 2.58, 1.45, 1.22, 9.68, 11.8, 9.68, 11.8, 2.7, 4.9, 9.68, 11.8, 2.7,  2.58,1.3]  # seconds
        encode_20000 = [6.2, 6.1, 3.1, 2.7, 20.64, 28, 20.64, 28, 6.2, 11, 20.64, 28, 6.1,6.2, 2.6]  # seconds

        #           V-Exo(small)  Z-Exo(med)  Edge-kla AAu(large) NJN RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 V-Exo(med) V-Exo(small) SOF-exo-HUge
        frame_200 = [0.44, 0.8, 0.5, 0.39, 2, 4, 2, 4, 0.44, 0.5, 2, 4, 0.8, 0.44, 0.77]
        frame_1500 = [1.8, 1.8, 2.46, 2.1, 11, 9.4, 11, 1.8, 2, 9.4, 11, 1.8,1.8,1.83]
        frame_3000 = [2.6, 2.5, 2.9, 3.8,14, 14, 14, 14, 2.6, 2.8, 14, 14,2.5, 2.6,2]
        frame_6500 = [9.4, 8.7, 9.9, 14.2, 55, 49, 55, 49, 9.4, 9.8, 55, 49, 8.7,9.4, 8.4]
        frame_20000 = [18.2, 18.5, 20.9, 31, 17, 112, 117, 112, 18.2, 21, 117, 112,18.5,18.2,17.5]

        # 		   V-Exo(small) Z-Exo(med) Edge-kla AAu(large) NJN   RPi4 NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 V-Exo(med) V-Exo(small) SOF-exo-HUge
        inference = [0.29, 0.26, 0.225, 0.282,1.94, 1.1, 1.94, 1.1, 0.29, 0.3, 1.94, 1.1, 0.26,0.29,0.246]
        lowtrain = [32, 26.5, 23, 17,152, 102, 152, 102, 32, 35, 152, 102,26.5, 32,19.6]  # seconds
        hightrain = [109, 103, 77, 33, 232, 467, 232, 467, 109, 115, 232, 467, 103, 109, 62.5]  # seconds

        # Converting string to list
        tasks0 = ["encode_20000", "frame_20000", "lowtrain", "hightrain", "inference", "transcode", "package"]  # sys.argv[1].strip("][").split(",")
        # print(type(tasks0))

        num_of_apps = 1
        newtasks = [0 for i in range(len(tasks0) * num_of_apps)]
        for k in range(num_of_apps):
            for i in range(len(tasks0)):
                newtasks[i + (k * len(tasks0))] = tasks0[i] + str(k)

        resources = ["vm-exo-SMALL", "ZHR-exo-med", "Edge-kla", "AAu(large)",
                     "jetson", "rpi4", "jetson", "rpi4", "vm-exo-SMALL", "vm-exo-Tiny", "jetson", "rpi4","vm-exo-med","vm-exo-SMALL",
                     "SOF-exo-HUge"]

        lat = [[0.5, 12.5, 7.3, 4.8, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5,0.5,0.5, 15],
            [12.5, 0.5, 6.7, 16.6, 23.6, 23.2, 23.6, 23.2, 23.2, 12.5, 12.5, 23.2, 23.2,12.5,12.5, 25.9],
            [7.3, 6.7, 0.5, 11.5,12.5, 12.6, 12.6, 12.6, 12.6, 7.3, 7.3, 23.2, 23.2,7.3,7.3, 21],
            [4.8, 16.6, 11.5, 0.5, 11.5, 12, 11.5, 12, 12, 4.8, 4.8, 12.6, 12.6, 12, 12,4.8,4.8, 10],
            [7.2, 23.6, 12.5, 11.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7.2, 7.2, 0.5, 0.5,7.2,7.2, 22.8],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5, 22.8],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5, 22.6],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5, 22.8],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5, 22.6],
            [0.5, 12.5, 7.3, 4.8,7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 15],
            [0.5, 12.5, 7.3, 4.8,7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 15],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5, 22.8],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5, 22.6],
            [0.5, 12.5, 7.3, 4.8, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5,0.5,0.5,15],
            [0.5, 12.5, 7.3, 4.8, 7.2, 7.5, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5,0.5,0.5,15],
            [15, 25.9, 21, 10, 22.8, 22.8, 22.8, 22.6, 22.6, 15, 15, 22.8, 22.6,15,15,0.5]]



        BW_r = [[13000, 1950, 5000, 950, 900, 900, 900, 900, 13000, 13000, 900, 900,13000, 13000, 1500],
             [1950, 1200, 1400, 900,  700, 770, 700, 770, 1950, 1950, 700, 770,1950,1950, 900],
             [5000, 1400, 12000, 930,  840, 850, 840, 850, 5000, 5000, 840, 850,5000,5000, 1200],
             [950, 900, 930, 930, 930, 850, 930, 850, 950, 950, 930, 850, 950, 950, 920],
             [900, 700, 840, 930, 920, 920, 920, 920, 900, 900, 920, 920, 900, 900, 900],
             [900, 770, 850, 850, 920, 850, 920, 850, 900, 900, 920, 920, 920, 850, 900, 900, 850],
             [900, 700, 840, 930, 920, 920, 920, 920, 900, 900, 920, 920, 900, 900, 900],
             [900, 770, 850, 850, 920, 850, 920, 850, 900, 900, 920, 850, 900, 900, 850],
             [13000, 1950, 5000, 900, 900, 900, 900, 900, 13000, 13000, 900, 900, 13000, 13000,1500],
             [13000, 1950, 5000, 900, 900, 900, 900, 900, 13000, 13000, 900, 900,13000, 13000, 1500],
             [900, 700, 840, 930, 920, 920, 920, 920, 900, 900, 920, 900,900, 900, 900],
             [900, 770, 850, 850, 920, 850, 920, 850, 900, 900, 900, 920,900, 900, 850],
             [13000, 1950, 5000, 950, 900, 900, 900, 900, 13000, 13000, 900, 900, 13000, 13000, 1500],
             [13000, 1950, 5000, 950, 900, 900, 900, 900, 13000, 13000, 900, 900, 13000, 13000, 1500],
             [500, 900, 1200, 920, 900, 850, 900, 850, 1500, 1500, 900, 850, 1500, 1500, 12000]]

        index_of_segment = 4
        seg_size = [286720, 2457600, 3440640, 14400000, 20971520]  # bits
        video_size = [2000000, 14000000, 28000000, 60000000, 204800000]  # bits
        
        T = [0 for i in range(len(tasks0))]
        Tm = [0 for i in range(len(tasks0))]
        Tr = [0 for i in range(len(tasks0))]

        if (task == "encode_20000"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "frame_20000"):
            ###print(frame_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = frame_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "hightrain"):
            # print(hightrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = hightrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "lowtrain"):
            # print(lowtrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = lowtrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "inference"):
            # print(inference[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = inference[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]))  
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "transcode"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "package"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        return (Tm, Tr, T)

    ###########################################################################################################################################
    # time interval 5
    ###########################################################################################################################################
    # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
    # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
    #             V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)

def comp_times_DeltaInterval_5_video(task, dev_new, dev1, dev2, dev3):
        #      V-Exo(small)  Z-Exo(med) Edge-kla AAu(large)  NJN   NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 V-Exo(med) V-Exo(small) V-Exo(small) V-Exo(Tiny) SOF-exo-HUge
        encode_200 = [0.44, 0.4, 0.28, 0.17, 1.9, 1.9, 2.16, 0.44, 0.46, 1.9, 2.16, 0.4, 0.44, 0.44, 0.46, 0.23]  # seconds
        encode_1500 = [0.65, 0.62, 0.5, 0.36, 2.63, 2.63, 3.19, 0.65, 1.14, 2.63, 3.19, 0.62, 0.65, 0.65, 1.14, 0.43]  # seconds
        encode_3000 = [0.9, 0.89, 0.63, 0.47, 3.48, 3.48, 4.4, 0.9, 1.6, 3.48, 4.4, 0.89, 0.9,0.9, 1.6, 0.55]  # seconds
        encode_6500 = [2.7, 2.58, 1.45, 1.22, 9.68, 9.68, 11.8, 2.7, 4.9, 9.68, 11.8, 2.7, 2.58,2.7, 4.9, 1.3]  # seconds
        encode_20000 = [6.2, 6.1, 3.1, 2.7, 20.64, 20.64, 28, 6.2, 11, 20.64, 28, 6.1, 6.2, 6.2, 11, 2.6]  # seconds

        #           V-Exo(small)  Z-Exo(med)  Edge-kla AAu(large) NJN NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 V-Exo(med) V-Exo(small) V-Exo(small) V-Exo(Tiny) SOF-exo-HUge
        frame_200 = [0.44, 0.8, 0.5, 0.39, 2, 2, 4, 0.44, 0.5, 2, 4, 0.8, 0.44,0.44, 0.5, 0.77]
        frame_1500 = [1.8, 1.8, 2.46, 2.1, 11,11, 1.8, 2, 9.4, 11, 1.8, 1.8,11, 9.4, 1.83]
        frame_3000 = [2.6, 2.5, 2.9, 3.8, 14, 14, 14, 2.6, 2.8, 14, 14, 2.5, 2.6,2.6, 2.8, 2]
        frame_6500 = [9.4, 8.7, 9.9, 14.2, 55, 55, 49, 9.4, 9.8, 55, 49, 8.7, 9.4,9.4, 9.8, 8.4]
        frame_20000 = [18.2, 18.5, 20.9, 31, 117, 117, 112, 18.2, 21, 117, 112, 18.5, 18.2, 18.2, 21, 17.5]

        # 		   V-Exo(small) Z-Exo(med) Edge-kla AAu(large) NJN   NJN   RPi4 V-Exo(small) V-Exo(tiny) NJN   RPi4 V-Exo(med) V-Exo(small) V-Exo(small) V-Exo(Tiny) SOF-exo-HUge
        inference = [0.29, 0.26, 0.225, 0.282,1.94, 1.94, 1.1, 0.29, 0.3, 1.94, 1.1, 0.26,0.29,0.29, 0,0.246]
        lowtrain = [32, 26.5, 23, 17,152, 152, 102, 32, 35, 152, 102,26.5, 32,32, 0, 19.6]  # seconds
        hightrain = [109, 103, 77, 33, 232,232, 467, 109, 115, 232, 467, 103, 109, 109, 0, 62.5]  # seconds

        # Converting string to list
        tasks0 = ["encode_20000", "frame_20000", "lowtrain", "hightrain", "inference", "transcode", "package"]  # sys.argv[1].strip("][").split(",")
        # print(type(tasks0))

        num_of_apps = 1
        newtasks = [0 for i in range(len(tasks0) * num_of_apps)]
        for k in range(num_of_apps):
            for i in range(len(tasks0)):
                newtasks[i + (k * len(tasks0))] = tasks0[i] + str(k)

        resources = ["vm-exo-SMALL", "ZHR-exo-med", "Edge-kla", "AAu(large)",
                     "jetson", "rpi4", "jetson", "vm-exo-SMALL", "vm-exo-Tiny", "jetson", "rpi4", "vm-exo-med",
                     "vm-exo-SMALL","vm-exo-SMALL", "vm-exo-Tiny",
                     "SOF-exo-HUge"]


        lat = [[0.5, 12.5, 7.3, 4.8, 7.2, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5,0.5,0.5,0.5, 0.5, 15],
            [12.5, 0.5, 6.7, 16.6, 23.6, 23.6, 23.2, 23.2, 12.5, 12.5, 23.2, 23.2,12.5,12.5,12.5,12.5, 25.9],
            [7.3, 6.7, 0.5, 11.5,12.5, 12.6, 12.6, 12.6, 7.3, 7.3, 23.2, 23.2,7.3,7.3,7.3,7.3, 21],
            [4.8, 16.6, 11.5, 0.5, 11.5, 11.5, 12, 12, 4.8, 4.8, 12.6, 12.6, 12, 12,4.8,4.8,4.8,4.8, 10],
            [7.2, 23.6, 12.5, 11.5, 0.5, 0.5, 0.5, 0.5, 7.2, 7.2, 0.5, 0.5,7.2,7.2, 7.2,7.2,22.8],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5, 7.5, 7.5,22.6],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5,7.5, 7.5, 22.8],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5,7.5, 7.5, 22.6],
            [0.5, 12.5, 7.3, 4.8,7.2, 7.5, 7.5, 7.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5,0.5, 0.5, 15],
            [0.5, 12.5, 7.3, 4.8,7.2, 7.5, 7.5, 7.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5,0.5, 0.5, 15],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5, 7.5, 7.5,22.8],
            [7.5, 23.2, 12.6, 12, 0.5, 0.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,7.5, 7.5,7.5, 7.5, 22.6],
            [0.5, 12.5, 7.3, 4.8, 7.2, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5,0.5,0.5,15],
            [0.5, 12.5, 7.3, 4.8, 7.2, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5,0.5,0.5,0.5,0.5,15],
            [0.5, 12.5, 7.3, 4.8, 7.2, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5,0.5, 0.5, 15],
            [0.5, 12.5, 7.3, 4.8, 7.2, 7.5, 7.5, 7.5, 0.5, 0.5, 7.5, 7.5, 0.5, 0.5, 0.5, 0.5,15],
            [15, 25.9, 21, 10, 22.8, 22.8, 22.6, 22.6, 15, 15, 22.8, 22.6,15,15,15,15,0.5]]



        BW_r = [[13000, 1950, 5000, 950, 900, 900, 900, 13000, 13000, 900, 900,13000, 13000,13000, 13000, 1500],
             [1950, 1200, 1400, 900,  700, 700, 770, 1950, 1950, 700, 770,1950,1950, 1950,1950, 900],
             [5000, 1400, 12000, 930,  840, 840, 850, 5000, 5000, 840, 850,5000,5000, 5000,5000,1200],
             [950, 900, 930, 930, 930, 930, 850, 950, 950, 930, 850, 950, 950,950, 950, 920],
             [900, 700, 840, 930, 920, 920, 920, 900, 900, 920, 920, 900, 900,900, 900,  900],
             [900, 700, 840, 930, 920, 920, 920, 900, 900, 920, 920, 900, 900,900, 900,  900],
             [900, 770, 850, 850, 920, 920, 850, 900, 900, 920, 850, 900, 900,900, 900, 850],
             [13000, 1950, 5000, 900, 900, 900, 900, 13000, 13000, 900, 900, 13000, 13000, 13000, 13000,1500],
             [13000, 1950, 5000, 900, 900, 900, 900, 13000, 13000, 900, 900,13000, 13000, 13000, 13000, 1500],
             [900, 700, 840, 930, 920, 920, 920, 900, 900, 920, 900,900, 900, 900, 900,900],
             [900, 770, 850, 850, 920, 920, 850, 900, 900, 900, 920,900, 900, 900, 900,850],
             [13000, 1950, 5000, 950, 900, 900, 900, 13000, 13000, 900, 900, 13000, 13000,13000, 13000, 1500],
             [13000, 1950, 5000, 950, 900, 900, 900, 13000, 13000, 900, 900, 13000, 13000, 13000, 13000,1500],
             [13000, 1950, 5000, 950, 900, 900, 900, 13000, 13000, 900, 900, 13000, 13000, 13000, 13000,1500],
             [13000, 1950, 5000, 950, 900, 900, 900, 13000, 13000, 900, 900, 13000, 13000,13000, 13000, 1500],
             [500, 900, 1200, 920, 900, 900, 850, 1500, 1500, 900, 850, 1500, 1500,1500, 1500, 12000]]

        index_of_segment = 4
        seg_size = [286720, 2457600, 3440640, 14400000, 20971520]  # bits
        video_size = [2000000, 14000000, 28000000, 60000000, 204800000]  # bits
        
        T = [0 for i in range(len(tasks0))]
        Tm = [0 for i in range(len(tasks0))]
        Tr = [0 for i in range(len(tasks0))]

        if (task == "encode_20000"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "frame_20000"):
            ###print(frame_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = frame_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "hightrain"):
            # print(hightrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = hightrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "lowtrain"):
            # print(lowtrain[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = lowtrain[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "inference"):
            # print(inference[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = inference[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]))  
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            # print (T[tasks0.index(task)][dev_new])

            
        elif (task == "transcode"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = max(((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                    ((0.000001) * (video_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        elif (task == "package"):
            ###print(encode_20000[dev_new]," ",type(rec))
            Tm[tasks0.index(task)] = encode_20000[dev_new]  # data size: 8sec video.
            Tr[tasks0.index(task)] = ((0.000001) * (video_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])
            T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
            #print (T[tasks0.index(task)])

            
        return (Tm, Tr, T)