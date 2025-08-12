#!/usr/bin/env bash
set -euo pipefail

# 1. Add NVIDIA's official CUDA keyring (for Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 2. Update Apt repositories
sudo apt-get update

# 3. Install CUDA 12.6 toolkit (+ recommended driver)
#    This also installs the default driver from the same repo that matches CUDA 12.6
sudo apt-get install -y cuda-toolkit-12-6

# 4. (Optional) If you specifically want a particular driver, e.g., 535 or 560:
#    sudo apt-get install -y nvidia-driver-535

# 5. Environment Variables for CUDA
#    Add them to /etc/profile.d or your ~/.bashrc for permanent usage:
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda_path.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda_path.sh
sudo chmod +x /etc/profile.d/cuda_path.sh

# 6. Done! Reboot or log out/in so new driver and environment are active
echo "Installation complete. Consider rebooting for new driver modules to load."

