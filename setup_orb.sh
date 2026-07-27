#!/bin/bash
set -e

echo "=== Installing dependencies ==="
# sudo apt-get update
# sudo apt-get install -y libglew-dev libegl1-mesa-dev libpython3-dev cmake git build-essential

cd /home/nalin/gps_denied_project

# Build Pangolin
if [ ! -d "Pangolin" ]; then
    echo "=== Building Pangolin ==="
    git clone https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    # Use an older commit of Pangolin that is compatible with ORB-SLAM3 (C++11/14)
    git checkout ad8b5f83222291c51b4800d5a58967b579afde4e || true
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    echo "Pangolin build completed. Proceeding to ORB-SLAM3..."
    cd ../..
else
    echo "Pangolin already exists."
fi

# Clone ORB-SLAM3
if [ ! -d "ORB_SLAM3" ]; then
    echo "=== Cloning ORB-SLAM3 ==="
    git clone https://github.com/Nalin959/ORB_SLAM3.git
    cd ORB_SLAM3
    
    # Give execute permissions to build scripts
    chmod +x build.sh
    chmod +x build_ros.sh
    
    # Building Thirdparty (DBoW2 and g2o)
    echo "=== Building ORB-SLAM3 Thirdparty ==="
    cd Thirdparty/DBoW2
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    cd ../../g2o
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    cd ../../../
    
    echo "=== Thirdparty built! Core compilation will be done next ==="
else
    echo "ORB_SLAM3 already exists."
fi
