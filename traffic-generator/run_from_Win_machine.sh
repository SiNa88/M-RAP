#!/bin/bash

START=$(date +%s%3N)

# Set a counter variable
counter=1
END=20
for i in $(seq 1 $END); do
        #wget http://194.182.172.55/index.html
        #wget http://194.182.172.55/category.html?tags=blue
        #wget http://194.182.172.55/customer-orders.html
        #wget http://194.182.172.55/basket.html
        C:/Users/narmehran/AppData/Local/Programs/Python/Python39/python.exe D:/00Research/00Fog/003-Zara/parallel-requests.py http://194.182.172.55/basket.html
done
END=$(date +%s%3N)
DIFF=$(( $END - $START ))
echo "It took $DIFF milliseconds" > sample.txt
#rm index.html*
#rm category.html*
#rm customer-orders.html*
#rm basket.html*
exit