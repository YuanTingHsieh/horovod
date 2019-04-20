# CS744 final project setup script

# First open an experiment in cloudlab

sudo apt-get update && sudo apt-get install -y python3-pip
sudo apt-get install -y python-cffi
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar -zxvf openmpi-4.0.1.tar.gz
cd openmpi-4.0.1/
./configure --prefix=/usr/local
make -j 4 all 
sudo make install
sudo ldconfig
#sudo -H python3 -m pip install tensorflow
python3 -m pip install tensorflow

# make sure you can access to different nodes using ssh

#sudo -H python3 -m pip install horovod
python3 -m pip install horovod

#git clone https://github.com/horovod/horovod.git

# to build horovod from source
git clone --recursive https://github.com/YuanTingHsieh/horovod.git
cd horovod
python3 setup.py sdist
python3 -m pip install dist/horovod*.tar.gz
# remember to modify bin/horovodrun first line python to python3
chmod 777 bin/horovodrun
cd ../

# be sure to gen sshkey and copy it inside cloud lab
# and make sure each node can ssh to one another

# to run
./horovod/bin/hordovodrun -np 3 -H node0:1,node1:1,node2:1 python3 ./horovod/examples/tensorflow_mnist.py
