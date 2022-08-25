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

def comp_times_DeltaInterval_0_ecommerce(task, dev_new, dev1, dev2, dev3):
        #####################################SockShop####################################
        #      V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
        mips = [7200 , 7100 , 3700, 7200, 7200 ,12000, 58000 , 21700 , 4080, 5100,  28800]
        
        mi_sockshop = [350,350,350,400,350,350,350,350]

        time_sockshop = [[0] * len(mips) for i in range(len(mi_sockshop))]
        for i in range(len(mi_sockshop)):
                for j in range(len(mips)):
                        time_sockshop[i][j] = (numpy.round(mi_sockshop[i]/mips[j],6))
                #print((time_sockshop[i]))


        # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
        # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
        ##################################################################################
        #time interval 0
        ##################################################################################
        time = [[0.048611, 0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.012153],
                [0.048611, 0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.012153],
                [0.048611, 0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.012153],
                [0.055556, 0.056338, 0.108108, 0.055556, 0.055556, 0.033333, 0.006897, 0.018433, 0.098039, 0.078431, 0.013889],
                [0.048611, 0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.012153],
                [0.048611, 0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.012153],
                [0.048611, 0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.012153],
                [0.048611, 0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.012153]]
        
        index_of_segment = 3
        data_size = [6.22*1024*8, 8.48*1024*8, 11.25*1024*8, 22.48*1024*8] # bits
        SIZE = 1

        # Converting string to list
        tasks0 = ["Web-UI", "Login", "Orders", "Shopping-cart", "Catalogue", "Accounts", "Payment", "Shipping"]  # sys.argv[1].strip("][").split(",")
        #print((tasks0.index(task)))

        #              0	       1	      2      		3      4        5        6       7		8      9      10
        resources = ["V-Exo(med)", "V-Exo(small)", "V-Exo(tiny)",  "Z-Exo(med)", "M-Exo(med)", "Edge-kla", "AAu(large)", "Lenovo",  "NJN",   "RPi4",   "Sofia-Exo(Huge)"]
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

        T = [1 for i in range(len(tasks0))]
        Tm = [1 for i in range(len(tasks0))]
        Tr = [1 for i in range(len(tasks0))]

        Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
        #print(Tm[tasks0.index(task)])
        #print(dev_new)
        Tr[tasks0.index(task)] = max(((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
        T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        return (Tm, Tr, T)
def comp_times_DeltaInterval_1_ecommerce(task, dev_new, dev1, dev2, dev3):
        #####################################SockShop####################################
        #      V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
        mips = [7200 , 7100 , 3700, 7200, 7200 ,12000, 58000 , 21700 , 4080, 5100,  28800]
        
        mi_sockshop = [350,350,350,400,350,350,350,350]

        time_sockshop = [[0] * len(mips) for i in range(len(mi_sockshop))]
        for i in range(len(mi_sockshop)):
                for j in range(len(mips)):
                        time_sockshop[i][j] = (numpy.round(mi_sockshop[i]/mips[j],6))
                #print((time_sockshop[i]))


        # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
        # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
        ##################################################################################
        # time interval 1
        ##################################################################################
        time = [[0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627,  0.085784, 0.068627,
                 0.012153],
                [0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784, 0.068627,
                 0.012153],
                [0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784, 0.068627,
                 0.012153],
                [0.056338, 0.108108, 0.055556, 0.055556, 0.033333, 0.006897, 0.018433, 0.098039, 0.078431, 0.098039, 0.078431,
                 0.013889],
                [0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784, 0.068627,
                 0.012153],
                [0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784, 0.068627,
                 0.012153],
                [0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784, 0.068627,
                 0.012153],
                [0.049296, 0.094595, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784, 0.068627,
                 0.012153]]

        index_of_segment = 3
        data_size = [6.22 * 1024 * 8, 8.48 * 1024 * 8, 11.25 * 1024 * 8, 22.48 * 1024 * 8]  # bits
        SIZE = 1

        # Converting string to list
        tasks0 = ["Web-UI", "Login", "Orders", "Shopping-cart", "Catalogue", "Accounts", "Payment",
                  "Shipping"]  # sys.argv[1].strip("][").split(",")
        # print((tasks0.index(task)))

        resources = ["V-Exo(small)", "V-Exo(tiny)", "Z-Exo(med)", "M-Exo(med)", "Edge-kla", "AAu(large)",
                     "Lenovo", "NJN", "RPi4", "NJN", "RPi4", "Sofia-Exo(Huge)"]
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

        T = [1 for i in range(len(tasks0))]
        Tm = [1 for i in range(len(tasks0))]
        Tr = [1 for i in range(len(tasks0))]

        Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
        #print(Tm[tasks0.index(task)])
        #print(dev_new)
        Tr[tasks0.index(task)] = max(((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
        T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        return (Tm, Tr, T)

def comp_times_DeltaInterval_2_ecommerce(task, dev_new, dev1, dev2, dev3):
        #####################################SockShop####################################
        #      V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
        mips = [7200 , 7100 , 3700, 7200, 7200 ,12000, 58000 , 21700 , 4080, 5100,  28800]
        
        mi_sockshop = [350,350,350,400,350,350,350,350]

        time_sockshop = [[0] * len(mips) for i in range(len(mi_sockshop))]
        for i in range(len(mi_sockshop)):
                for j in range(len(mips)):
                        time_sockshop[i][j] = (numpy.round(mi_sockshop[i]/mips[j],6))
                #print((time_sockshop[i]))


        # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
        # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
        ##################################################################################
        # time interval 2
        ##################################################################################
        time = [[0.049296, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627,0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,
                 0.012153],
                [0.056338, 0.055556, 0.055556, 0.033333, 0.006897, 0.018433, 0.098039, 0.078431, 0.098039,0.078431, 0.056338, 0.108108,
                 0.013889],
                [0.049296, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627,0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784, 0.068627, 0.049296, 0.094595,
                 0.012153]]

        index_of_segment = 3
        data_size = [6.22 * 1024 * 8, 8.48 * 1024 * 8, 11.25 * 1024 * 8, 22.48 * 1024 * 8]  # bits
        SIZE = 1

        # Converting string to list
        tasks0 = ["Web-UI", "Login", "Orders", "Shopping-cart", "Catalogue", "Accounts", "Payment",
                  "Shipping"]  # sys.argv[1].strip("][").split(",")
        # print((tasks0.index(task)))

        #              0	       1	      2      		3      4        5        6       7		8      9      10
        resources = ["V-Exo(small)", "Z-Exo(med)", "M-Exo(med)", "Edge-kla", "AAu(large)",
                     "Lenovo", "NJN", "RPi4", "NJN", "RPi4","V-Exo(small)", "V-Exo(tiny)", "Sofia-Exo(Huge)"]
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

        T = [1 for i in range(len(tasks0))]
        Tm = [1 for i in range(len(tasks0))]
        Tr = [1 for i in range(len(tasks0))]

        Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
        #print(Tm[tasks0.index(task)])
        #print(dev_new)
        Tr[tasks0.index(task)] = max(((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
        T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        return (Tm, Tr, T)

def comp_times_DeltaInterval_3_ecommerce(task, dev_new, dev1, dev2, dev3):
        #####################################SockShop####################################
        #      V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
        mips = [7200 , 7100 , 3700, 7200, 7200 ,12000, 58000 , 21700 , 4080, 5100,  28800]
        
        mi_sockshop = [350,350,350,400,350,350,350,350]

        time_sockshop = [[0] * len(mips) for i in range(len(mi_sockshop))]
        for i in range(len(mi_sockshop)):
                for j in range(len(mips)):
                        time_sockshop[i][j] = (numpy.round(mi_sockshop[i]/mips[j],6))
                #print((time_sockshop[i]))


        # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
        # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
        ##################################################################################
        # time interval 3
        ##################################################################################
        time = [[0.049296, 0.048611,0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627,0.049296, 0.094595,  0.085784,0.068627,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,  0.085784,0.068627,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,0.085784,0.068627,
                 0.012153],
                [0.056338, 0.055556, 0.033333, 0.006897, 0.018433, 0.098039, 0.078431, 0.098039,0.078431, 0.056338, 0.108108, 0.098039,0.078431,
                 0.013889],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627,0.049296, 0.094595, 0.085784,0.068627,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,  0.085784,0.068627,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595, 0.085784,0.068627,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.016129, 0.085784, 0.068627, 0.085784, 0.068627, 0.049296, 0.094595,  0.085784, 0.068627,
                 0.012153]]

        index_of_segment = 3
        data_size = [6.22 * 1024 * 8, 8.48 * 1024 * 8, 11.25 * 1024 * 8, 22.48 * 1024 * 8]  # bits
        SIZE = 1

        # Converting string to list
        tasks0 = ["Web-UI", "Login", "Orders", "Shopping-cart", "Catalogue", "Accounts", "Payment",
                  "Shipping"]  # sys.argv[1].strip("][").split(",")
        # print((tasks0.index(task)))

        #              0	       1	      2      		3      4        5        6       7		8      9      10
        resources = ["V-Exo(small)", "Z-Exo(med)", "Edge-kla", "AAu(large)",
                     "Lenovo", "NJN", "RPi4", "NJN", "RPi4","V-Exo(small)", "V-Exo(tiny)","NJN", "RPi4", "Sofia-Exo(Huge)"]
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

        T = [1 for i in range(len(tasks0))]
        Tm = [1 for i in range(len(tasks0))]
        Tr = [1 for i in range(len(tasks0))]

        Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
        #print(Tm[tasks0.index(task)])
        #print(dev_new)
        Tr[tasks0.index(task)] = max(((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
        T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        return (Tm, Tr, T)

def comp_times_DeltaInterval_4_ecommerce(task, dev_new, dev1, dev2, dev3):
        #####################################SockShop####################################
        #      V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
        mips = [7200 , 7100 , 3700, 7200, 7200 ,12000, 58000 , 21700 , 4080, 5100,  28800]
        
        mi_sockshop = [350,350,350,400,350,350,350,350]

        time_sockshop = [[0] * len(mips) for i in range(len(mi_sockshop))]
        for i in range(len(mi_sockshop)):
                for j in range(len(mips)):
                        time_sockshop[i][j] = (numpy.round(mi_sockshop[i]/mips[j],6))
                #print((time_sockshop[i]))


        # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
        # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
        ##################################################################################
        # time interval 4
        ##################################################################################
        time = [[0.049296, 0.048611,0.029167, 0.006034, 0.085784, 0.068627, 0.085784,0.068627,0.049296, 0.094595,  0.085784,0.068627, 0.048611, 0.049296,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,  0.085784,0.068627, 0.048611, 0.049296,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034,0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,0.085784,0.068627, 0.048611, 0.049296,
                 0.012153],
                [0.056338, 0.055556, 0.033333, 0.006897, 0.098039, 0.078431, 0.098039,0.078431, 0.056338, 0.108108, 0.098039,0.078431, 0.055556, 0.056338,
                 0.013889],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.068627, 0.085784,0.068627,0.049296, 0.094595, 0.085784,0.068627, 0.048611, 0.049296,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595,  0.085784,0.068627, 0.048611, 0.049296,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.068627, 0.085784,0.068627, 0.049296, 0.094595, 0.085784,0.068627, 0.048611, 0.049296,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.068627, 0.085784, 0.068627, 0.049296, 0.094595,  0.085784, 0.068627, 0.048611, 0.049296,
                 0.012153]]


        index_of_segment = 3
        data_size = [6.22 * 1024 * 8, 8.48 * 1024 * 8, 11.25 * 1024 * 8, 22.48 * 1024 * 8]  # bits
        SIZE = 1

        # Converting string to list
        tasks0 = ["Web-UI", "Login", "Orders", "Shopping-cart", "Catalogue", "Accounts", "Payment",
                  "Shipping"]  # sys.argv[1].strip("][").split(",")
        # print((tasks0.index(task)))

        #              0	       1	      2      		3      4        5        6       7		8      9      10
        resources = ["V-Exo(small)", "Z-Exo(med)", "Edge-kla", "AAu(large)","NJN", "RPi4", "NJN", "RPi4","V-Exo(small)", "V-Exo(tiny)","NJN", "RPi4", "V-Exo(med)", "V-Exo(small)","Sofia-Exo(Huge)"]
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

        T = [1 for i in range(len(tasks0))]
        Tm = [1 for i in range(len(tasks0))]
        Tr = [1 for i in range(len(tasks0))]

        Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
        #print(Tm[tasks0.index(task)])
        #print(dev_new)
        Tr[tasks0.index(task)] = max(((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
        T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        return (Tm, Tr, T)

def comp_times_DeltaInterval_5_ecommerce(task, dev_new, dev1, dev2, dev3):
        #####################################SockShop####################################
        #      V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
        mips = [7200 , 7100 , 3700, 7200, 7200 ,12000, 58000 , 21700 , 4080, 5100,  28800]
        
        mi_sockshop = [350,350,350,400,350,350,350,350]

        time_sockshop = [[0] * len(mips) for i in range(len(mi_sockshop))]
        for i in range(len(mi_sockshop)):
                for j in range(len(mips)):
                        time_sockshop[i][j] = (numpy.round(mi_sockshop[i]/mips[j],6))
                #print((time_sockshop[i]))


        # addednodes=[[jetson,RPi4],[W_S,W_t],[jetson,RPi4],[W_M,W_S]],[W_S,W_t]]
        # del_nodes=[0(W_m),1(W_t),2(MUC_M),4(aau_M),5(RPi4)]
        ##################################################################################
        # time interval 5
        ##################################################################################
        time = [[0.049296, 0.048611,0.029167, 0.006034, 0.085784,0.085784,0.068627,0.049296, 0.094595,  0.085784,0.068627, 0.048611, 0.049296, 0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.085784,0.068627, 0.049296, 0.094595,  0.085784,0.068627, 0.048611, 0.049296, 0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034,0.085784, 0.085784,0.068627, 0.049296, 0.094595,0.085784,0.068627, 0.048611, 0.049296,  0.049296, 0.094595,
                 0.012153],
                [0.056338, 0.055556, 0.033333, 0.006897, 0.098039, 0.098039,0.078431, 0.056338, 0.108108, 0.098039,0.078431, 0.055556, 0.056338, 0.056338, 0.108108,
                 0.013889],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.085784,0.068627,0.049296, 0.094595, 0.085784,0.068627, 0.048611, 0.049296, 0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.085784,0.068627, 0.049296, 0.094595,  0.085784,0.068627, 0.048611, 0.049296, 0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.085784,0.068627, 0.049296, 0.094595, 0.085784,0.068627, 0.048611, 0.049296, 0.049296, 0.094595,
                 0.012153],
                [0.049296, 0.048611, 0.029167, 0.006034, 0.085784, 0.085784, 0.068627, 0.049296, 0.094595,  0.085784, 0.068627, 0.048611, 0.049296, 0.049296, 0.094595,
                 0.012153]]

        index_of_segment = 3
        data_size = [6.22 * 1024 * 8, 8.48 * 1024 * 8, 11.25 * 1024 * 8, 22.48 * 1024 * 8]  # bits
        SIZE = 1

        # Converting string to list
        tasks0 = ["Web-UI", "Login", "Orders", "Shopping-cart", "Catalogue", "Accounts", "Payment",
                  "Shipping"]  # sys.argv[1].strip("][").split(",")
        # print((tasks0.index(task)))

        #              0	       1	      2      		3      4        5        6       7		8      9      10
        resources = ["V-Exo(small)", "Z-Exo(med)", "Edge-kla", "AAu(large)","NJN", "RPi4", "NJN", "RPi4","V-Exo(small)", "V-Exo(tiny)","NJN", "RPi4", "V-Exo(med)", "V-Exo(small)", "V-Exo(small)", "V-Exo(tiny)","Sofia-Exo(Huge)"]
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

        T = [1 for i in range(len(tasks0))]
        Tm = [1 for i in range(len(tasks0))]
        Tr = [1 for i in range(len(tasks0))]

        Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
        #print(Tm[tasks0.index(task)])
        #print(dev_new)
        Tr[tasks0.index(task)] = max(((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
        T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        return (Tm, Tr, T)
