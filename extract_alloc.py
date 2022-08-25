import numpy
import json
import sys
sys.path.append('./diff_times')
from diff_times import  comp_times
sys.path.append('./diff_times_ecommerce')
from diff_times_ecommerce import  comp_times_ecommerce

res_num=11
app_num=10
for tt in range(1):
    alloc_file = "D:\\00Research\\00Fog\\004-Zara\\allocDefinition" + str(tt) + ".json"
    with open(alloc_file, "r") as json_file:
        content_app = json.load(json_file)
    #print(len(content_app['initialAllocation']))
    app1 = ["Web-UI", "Login", "Orders", "Shopping-cart", "Catalogue", "Accounts", "Payment", "Shipping"]
    service_num_app1=8
    T_1 = [[[[0] for k in range(res_num)] for i in range(service_num_app1)] for j in range(app_num)]
    Tm_1 = [[[[0] for k in range(res_num)] for i in range(service_num_app1)] for j in range(app_num)]
    Tr_1 = [[[[0] for k in range(res_num)] for i in range(service_num_app1)] for j in range(app_num)]
    dev_1 = [[[0] for i in range(service_num_app1)] for j in range(app_num)]
    compl_time = [0 for k in range(app_num)]

    app2 = ["encode_20000", "frame_20000", "lowtrain", "hightrain", "inference", "transcode", "package"]
    service_num_app2=7
    T_2 = [[[[0] for k in range(res_num)] for i in range(service_num_app2)] for j in range(app_num)]
    Tm_2 = [[[[0] for k in range(res_num)] for i in range(service_num_app2)] for j in range(app_num)]
    Tr_2 = [[[[0] for k in range(res_num)] for i in range(service_num_app2)] for j in range(app_num)]
    dev_2 = [[[0] for i in range(service_num_app2)] for j in range(app_num)]
    #compl_time_2 = 0
    for service in range(len(content_app['initialAllocation'])):

        if (service < 40):
            m=content_app['initialAllocation'][service]["module_name"]
            part1 = int(m.split('_')[1])
            task =(part1) % service_num_app1
            #print(content_app['initialAllocation'][service])
            #print(app1[task])
            #print(task)
            
            app = int(content_app['initialAllocation'][service]["app"])
            dev_1[app][task] = content_app['initialAllocation'][service]["id_resource"]
            if dev_1[app][task]==1000:
                dev_1[app][task]=10
            #print(dev_1[app][task])
            if (task == 0):
                Tm_1[app][task], Tr_1[app][task], T_1[app][task] = comp_times_ecommerce(app1[task],dev_1[app][task], 7, 7, 7)
                compl_time[app] +=  T_1[app][task][task] #+ max(T_1[app][task][task],T_1[app][task][task],T_1[app][task][task],T_1[app][task][task])
            elif(task == 1):
                Tm_1[app][task], Tr_1[app][task], T_1[app][task] = comp_times_ecommerce(app1[task],dev_1[app][task], dev_1[app][task-1], dev_1[app][task-1], dev_1[app][task-1])
                compl_time[app] +=  T_1[app][task][task] + max(T_1[app][task-1][task-1],T_1[app][task-1][task-1],T_1[app][task-1][task-1])
            elif(task == 2):
                Tm_1[app][task], Tr_1[app][task], T_1[app][task] = comp_times_ecommerce(app1[task],dev_1[app][task], dev_1[app][task-2], dev_1[app][task-2], dev_1[app][task-2])
                compl_time[app] +=  T_1[app][task][task] + max(T_1[app][task-2][task-2],T_1[app][task-2][task-2],T_1[app][task-2][task-2])
            elif(task == 3):
                Tm_1[app][task], Tr_1[app][task], T_1[app][task] = comp_times_ecommerce(app1[task],dev_1[app][task], dev_1[app][task-1], dev_1[app][task-3], dev_1[app][task-3])
                compl_time[app] +=  T_1[app][task][task] + max(T_1[app][task-1][task-1],T_1[app][task-3][task-3],T_1[app][task-3][task-3])
            elif(task == 4):
                Tm_1[app][task], Tr_1[app][task], T_1[app][task] = comp_times_ecommerce(app1[task],dev_1[app][task], dev_1[app][task-4], dev_1[app][task-4], dev_1[app][task-4])
                compl_time[app] +=  T_1[app][task][task] + max(T_1[app][task-4][task-4],T_1[app][task-4][task-4],T_1[app][task-4][task-4])
            elif(task == 5):
                Tm_1[app][task], Tr_1[app][task], T_1[app][task] = comp_times_ecommerce(app1[task],dev_1[app][task], dev_1[app][task-3], dev_1[app][task-4], dev_1[app][task-5])
                compl_time[app] +=  T_1[app][task][task] + max(T_1[app][task-3][task-3],T_1[app][task-4][task-4],T_1[app][task-5][task-5])
            elif(task == 6):
                Tm_1[app][task], Tr_1[app][task], T_1[app][task] = comp_times_ecommerce(app1[task],dev_1[app][task], dev_1[app][task-4], dev_1[app][task-4], dev_1[app][task-4])
                compl_time[app] +=  T_1[app][task][task] + max(T_1[app][task-4][task-4],T_1[app][task-4][task-4],T_1[app][task-4][task-4])
            elif(task == 7):
                Tm_1[app][task], Tr_1[app][task], T_1[app][task] = comp_times_ecommerce(app1[task],dev_1[app][task], dev_1[app][task-5], dev_1[app][task-5], dev_1[app][task-5])
                compl_time[app] +=  T_1[app][task][task] + max(T_1[app][task-5][task-5],T_1[app][task-5][task-5],T_1[app][task-5][task-5])

            #dev_1[app][task] = content_app['initialAllocation'][service]["id_resource"]
            #if dev_1[app][task]==1000:
            #    #dev_1[app][task]=10
            #    print(app," ",task," ",10," ",T_1[app][task][10])
            #else:
            print(app," ",task," ",dev_1[app][task]," ",T_1[app][task][task])
        else:
            m=content_app['initialAllocation'][service]["module_name"]
            part1 = int(m.split('_')[1])
            task =( part1 - 40) % service_num_app2
            #print(content_app['initialAllocation'][service])
            #print(app2[task])
            #print(t1)
            
            app = int(content_app['initialAllocation'][service]["app"])
            if (task == 0):
                Tm_2[app][task], Tr_2[app][task], T_2[app][task] = comp_times(app2[task], 7, 7, 7)
                compl_time[app] +=  T_2[app][task][task]
                #print(compl_time[app])
            elif(task == 4):
                Tm_2[app][task], Tr_2[app][task], T_2[app][task] = comp_times(app2[task], dev_2[app][task-1], dev_2[app][task-2], dev_2[app][task-2])
                compl_time[app] +=  T_2[app][task][task] + max(T_2[app][task-1][task-1],T_2[app][task-2][task-2],T_2[app][task-2][task-2])
                #print(compl_time[app])                
            elif(task == 5):
                Tm_2[app][task], Tr_2[app][task], T_2[app][task] = comp_times(app2[task], dev_2[app][task-1], dev_2[app][task-2], dev_2[app][task-3])
                compl_time[app] +=  T_2[app][task][task] + max(T_2[app][task-1][task-1],T_2[app][task-2][task-2],T_2[app][task-3][task-3])
                #print(compl_time[app])
            else:
                Tm_2[app][task], Tr_2[app][task], T_2[app][task] = comp_times(app2[task], dev_2[app][task-1], dev_2[app][task-1], dev_2[app][task-1])
                compl_time[app] +=  T_2[app][task][task] + max(T_2[app][task-1][task-1],T_2[app][task-1][task-1],T_2[app][task-1][task-1])
                #print(compl_time[app])

            dev_2[app][task] = content_app['initialAllocation'][service]["id_resource"]
            if dev_2[app][task]==1000:
                dev_2[app][task]=10
                print(app," ",task," ",10," ",T_2[app][task][10])
            #else:
                print(app," ",task," ",dev_2[app][task]," ",T_2[app][task][task])

#{"module_name": "0_0", "app": "0", "id_resource": 6},
'''print(0," ",numpy.round(compl_time[0],4))
print(1," ",numpy.round(compl_time[1],4))
print(2," ",numpy.round(compl_time[2],4))
print(3," ",numpy.round(compl_time[3],4))
print(4," ",numpy.round(compl_time[4],4))
print(5," ",numpy.round(compl_time[5],4))
print(6," ",numpy.round(compl_time[6],4))
print(7," ",numpy.round(compl_time[7],4))
print(8," ",numpy.round(compl_time[8],4))
print(9," ",numpy.round(compl_time[9],4))'''