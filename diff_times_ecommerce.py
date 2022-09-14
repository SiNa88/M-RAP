import numpy
import subprocess
import os
import json
import time
import sys
import random


def comp_times_ecommerce(task,dev_new, dev1, dev2, dev3):
        #####################################SockShop####################################
        #      V-Exo(med) V-Exo(small) V-Exo(tiny)  V-Exo(med) V-Exo(med) Edge-kla AAu(large) Lenovo  NJN   RPi4   V-Exo(med)
        mips = [7200 , 7100 , 3700, 7200, 7200 ,12000, 58000 , 21700 , 4080, 5100,  28800]
        
        mi_sockshop = [350,350,350,400,350,350,350,350]

        time_sockshop = [[0] * len(mips) for i in range(len(mi_sockshop))]
        for i in range(len(mi_sockshop)):
                for j in range(len(mips)):
                        time_sockshop[i][j] = (numpy.round(mi_sockshop[i]/mips[j],6))
                #print((time_sockshop[i]))

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

        #              0	       1	      2      		3               4        5              6            7	      8         9            10
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
        #print (dev_new)
        Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
        #print(Tm[tasks0.index(task)])
        Tr[tasks0.index(task)] = max(((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                                        ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
        T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        '''
        # print (type(Tm))
        # print (type(Tr))
        if (task == "Web-UI"):
                Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
                Tr[tasks0.index(task)] = ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
                T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        
        elif (task == "Login"):
                Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
                Tr[tasks0.index(task)] = ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
                T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)

        elif (task == "Orders"):
                Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
                Tr[tasks0.index(task)] = ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
                T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)

        elif (task == "Shopping-cart"):
                Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
                Tr[tasks0.index(task)] = ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
                T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)

        elif (task == "Catalogue"):
                # print(inference[0]," ",type(rec))
                Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
                Tr[tasks0.index(task)] = ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
                T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
                #Tr[0] = max(((0.000000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0]),
                #        ((0.000000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][0]) + (lat[dev2][0]))  
                # print (T[0][0])

        elif (task == "Accounts"):
                ###print(encode_20000[0]," ",type(rec))
                Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
                Tr[tasks0.index(task)] = max(((0.000000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new]),
                                        ((0.000000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][dev_new]) + (lat[dev2][dev_new]),
                                        ((0.000000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][dev_new]) + (lat[dev3][dev_new]))
                T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)

                #Tr[0] = max(((0.000000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][0]) + (lat[dev1][0]),
                #        ((0.000000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev2][0]) + (lat[dev2][0]),
                #        ((0.000000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev3][0]) + (lat[dev3][0]))
                #print (T[0])


        elif (task == "Payment"):
                Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
                Tr[tasks0.index(task)] = ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
                T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)

        elif (task == "Shipping"):
                Tm[tasks0.index(task)] = time[tasks0.index(task)][dev_new]  # data size: 8sec video.
                Tr[tasks0.index(task)] = ((0.000001) * (SIZE*data_size[index_of_segment]) / BW_r[dev1][dev_new]) + (lat[dev1][dev_new])        
                T[tasks0.index(task)] = numpy.round(numpy.round(Tm[tasks0.index(task)], 4) +  numpy.round(Tr[tasks0.index(task)], 4), 4)
        '''
        return (Tm, Tr, T)
#comp_times_ecommerce(0,0,0,0,0)