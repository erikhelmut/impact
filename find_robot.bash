#!/bin/bash

# Subnet to scan
SUBNET="10.90.90"

echo "Pinging all hosts in ${SUBNET}.0/24..."

# Loop through all IPs from 10.90.90.1 to 10.90.90.254
for i in {1..254}; do
    IP="${SUBNET}.${i}"
    ping -c 1 -W 1 $IP > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Host $IP is UP"
    fi
done
