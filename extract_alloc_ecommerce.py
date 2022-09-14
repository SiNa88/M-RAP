import json
import sys
sys.path.append('./diff_times_ecommerce')
from diff_times_ecommerce import  comp_times_ecommerce

res_num=11
service_num=8
app_num=10
app1 = ["Web-UI", "Login", "Orders", "Shopping-cart", "Catalogue", "Accounts", "Payment", "Shipping"]
for tt in range(1):
    alloc_file = "D:\\00Research\\00Fog\\003-Zara\\allocDefinition" + str(tt) + ".json"
    with open(alloc_file, "r") as json_file:
        content_app = json.load(json_file)

    #print(len(content_app['initialAllocation']))

    T = [[[[0] for k in range(res_num)] for i in range(service_num)] for j in range(app_num)]
    Tm = [[[[0] for k in range(res_num)] for i in range(service_num)] for j in range(app_num)]
    Tr = [[[[0] for k in range(res_num)] for i in range(service_num)] for j in range(app_num)]
    dev = [[[0] for i in range(service_num)] for j in range(app_num)]

    for service in range(40):
        m=content_app['initialAllocation'][service]["module_name"]
        part1 = int(m.split('_')[1])
        task =( part1) % 8
        #print(content_app['initialAllocation'][service])
        #print(app1[task])
        #print(t1)
        
        app = int(content_app['initialAllocation'][service]["app"])
        if (task == 0):
            Tm[app][task], Tr[app][task], T[app][task] = comp_times_ecommerce(app1[task], 7, 7, 7)
        elif(task == 4):
            Tm[app][task], Tr[app][task], T[app][task] = comp_times_ecommerce(app1[task], dev[app][task-1], dev[app][task-2], dev[app][task-2])
        elif(task == 5):
            Tm[app][task], Tr[app][task], T[app][task] = comp_times_ecommerce(app1[task], dev[app][task-1], dev[app][task-2], dev[app][task-3])
        else:
            Tm[app][task], Tr[app][task], T[app][task] = comp_times_ecommerce(app1[task], dev[app][task-1], dev[app][task-1], dev[app][task-1])

        dev[app][task] = content_app['initialAllocation'][service]["id_resource"]
        if dev[app][task]==1000:
            dev[app][task]=10
            print(app," ",task," ",10," ",T[app][task][10])
        else:
            print(app," ",task," ",content_app['initialAllocation'][service]["id_resource"]," ",T[app][task][content_app['initialAllocation'][service]["id_resource"]])

#{"module_name": "0_0", "app": "0", "id_resource": 6},