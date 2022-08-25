#!/bin/bash
#sudo tc qdisc add dev enp0s31f6 root netem delay 70ms
START=$(date +%s%3N)

# Set a counter variable
counter=1
END=2000
for i in $(seq 1 $END); do
	#wget http://194.182.172.55/index.html
	#wget http://194.182.172.55/category.html?tags=blue
	#wget http://194.182.172.55/customer-orders.html
	#wget http://194.182.172.55/basket.html
	#wget http://194.182.172.55/detail.html?id=3395a43e-2d88-40de-b95f-e00e1502085b
	####python3 parallel-requests.py http://194.182.172.55/basket.html
	
done
END=$(date +%s%3N)
DIFF=$(( $END - $START ))
echo "It took $DIFF milliseconds"
#rm index.html*
#rm category.html*
#rm customer-orders.html*
#rm basket.html*
#sudo tc qdisc del dev enp0s31f6 root netem delay 70ms
exit
