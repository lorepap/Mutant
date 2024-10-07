#!/bin/bash

base_path=$(pwd)/src
cd $base_path/kernel || exit 1
echo ''

# Build kernel module
echo '--- Building kernel module file ---'
echo ''
sudo make clean
sudo make

# Check if module exists and remove it
if lsmod | grep -q "mutant"; then
    echo "--- Removing existing mutant module ---"
    sudo rmmod mutant
fi
# Insert the newly built kernel module
echo '--- Inserting newly built kernel module ---'
echo ''
# Uncomment the following line if you need to sign the module
# sudo /usr/src/linux-$(uname -r)/scripts/sign-file sha256 ./key/MOK.priv ./key/MOK.der mutant.ko
sudo insmod mutant.ko

sudo sysctl -w net.ipv4.ip_forward=1

echo ''
# Set mutant as congestion control protocol
echo '-- Set mutant as congestion control protocol'
echo ''
sudo sysctl net.ipv4.tcp_congestion_control=mutant || exit 1