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
python3 -m pip install tensorflow

# make sure you can access to different nodes using ssh

# To install horovod (official build) just do python3 -m pip install horovod
# to build horovod from source
# to build for tensorflow is as below
# to build for pytorch need to add flag HOROVOD_WITH_PYTORCH=1 in front of python3
git clone --recursive https://github.com/YuanTingHsieh/horovod.git
cd horovod
python3 setup.py sdist
python3 -m pip install dist/horovod*.tar.gz
# remember to modify bin/horovodrun first line python to python3
chmod 777 bin/horovodrun
cd ../

# be sure to gen sshkey and copy it inside cloud lab
# and make sure each node can ssh to one another

# to run just horovod
./horovod/bin/hordovodrun -np 3 -H node0:1,node1:1,node2:1 python3 ./horovod/examples/tensorflow_mnist.py

# to run submitjob.py, move it outside, then run following commands
python submitjob.py --hosts node0 node1 node2 --cpus 6 6 6 --jobfile ./horovod/examples/pytorch_mnist.py --epochs 15

# to interrupt(ie to change resource)
echo "num" | nc node0 5000

# to remove 3 computing resources from all machines, just do the following
echo "3" | nc node0 5000

# the log can be found in horovodrun_0.log, horovodrun_1.log and so on
