import logging
import threading
import time
import socket
import sys
import argparse
import subprocess
import os
import signal
import random

#globals: TODO: refactor to not use global variables
#not sure how that would work with multithreading though, need to look into it
host_lst = None
cpu_lst = None
horovod_proc = None
jobfile = None
times_started = 0 #used for logging filename
restarting = False

aliases = {"10.10.1.2":"node1", "10.10.1.3":"node2", "127.0.0.1":"node0"}
ports = [5000, 5001, 5002] #in case we just ran and port not ready yet, provide some backup ports


#removes more resources until we get an even batches-per-allreduce
def removeAdditionalResources(old_total):

    global host_lst
    global cpu_lst
    
    #this loop runs once for each additional resource to remove
    while(sum(cpu_lst) > 0 and old_total % sum(cpu_lst) != 0):

        #take from a random machine for fairness purposes
        #but who really cares
        host_ind = random.randint(0, len(host_lst)-1)
        cpu_lst[host_ind] -= 1
        if cpu_lst[host_ind] <= 0:
            cpu_lst.pop(host_ind)
            host_lst.pop(host_ind)

    #if we've removed all resources, we weren't able to get a good allocation
    if sum(cpu_lst) <= 0:
        return False
    else:
        return True

    
#kills the current-running horovod instance
def killHorovod():
    global horovod_proc
    os.killpg(os.getpgid(horovod_proc.pid), signal.SIGTERM)
   

#executes the provided command as a subprocess
def startHorovod(cmd):

    global times_started
    global horovod_proc
    
    print("starting horovod with command: " + cmd)
    times_started += 1    
    horovod_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)


#given the relevant info provided by the user, forms it into the correct horovod format and returns it
def formHorovodCommand(machines, cpus, jobfile, batch_count, epochs):

    global times_started
    
    machine_resources = []
    for index in range(len(machines)):
        machine_resources.append(machines[index] + ":" + str(cpus[index]))
    command_str = "./horovod/bin/horovodrun "
    command_str += "-np " + str(sum(cpus)) + " -H "
    for index in range(len(machine_resources)):
        command_str += machine_resources[index]
        
        if index != len(machine_resources)-1:
            command_str += ","
            
    command_str += " --verbose 2 python3 " + jobfile + " --loadcp --epochs " + str(epochs) +" --batches " + \
                   str(int(batch_count)) + " > horovodrun_" + str(times_started) + ".out"        
    return command_str
                                

#This method runs as a daemon to listen for messages from other nodes
def listener():

    global host_lst
    global cpu_lst
    global ports
    global aliases
    global restarting

    s = socket.socket()
    success = False
    while(not success):
        for socket_port in ports:
            try:
                #Initalize the socket
                s.bind(('', socket_port))
                s.listen(5)
                success = True
                break
            except:
                pass

    if(not success):
        print("Failed to bind to a socket")
        exit(0)
    
    #Listen for incoming requests 
    print("Listening for messages...")
    while(True):
        conn, addr = s.accept()

        try:
            msg = int(conn.recv(1024))
        except:
            #malformed msg, just drop it
            conn.close()
            continue

        old_total_resources = sum(cpu_lst)
        for index in range(len(host_lst)):
            if aliases[str(addr[0])] == host_lst[index]:
                cpu_lst[index] = max(0, cpu_lst[index] - msg)

                #if # cpus is 0, remove the machine entirely
                if cpu_lst[index] == 0:
                    cpu_lst.pop(index)
                    host_lst.pop(index)
                    
                break
            
        restarting = True
        new_total_resources = sum(cpu_lst)
        revoked_resources = False

        if old_total_resources % new_total_resources != 0:
            revoked_resources = removeAdditionalResources(old_total_resources)
            if not revoked_resources:
                print("ERROR: change in resource profile requested by " + str(addr[0]) + " doesn't result in a whole number " +
                      "for batches-per-allreduce. Exiting..")
                exit(0)
            new_total_resources = sum(cpu_lst)

        if revoked_resources:
            print("INFO: Had to revoke additional resources to satisfy the request from client.")
            
        print("INFO: Resource update request received from " + str(addr[0]) + \
              "\n\tCPUs reduced from " + str(old_total_resources) + "->" + \
              str(new_total_resources) + "\n\tUpdated batches-per-allreduce: " + \
              str(old_total_resources/new_total_resources))
            
        
        killHorovod()
        startHorovod(formHorovodCommand(host_lst, cpu_lst, jobfile, old_total_resources/new_total_resources, args.epochs))
        restarting = False
        conn.close()

#MAIN
parser = argparse.ArgumentParser()
parser.add_argument("--hosts", nargs='+', type=str)
parser.add_argument("--cpus", nargs='+', type=int)
parser.add_argument("--jobfile", type=str)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()
try:
    host_lst = list(args.hosts)
    cpu_lst = list(args.cpus)
    jobfile = args.jobfile
    epochs = args.epochs
except:
    print("Bad args. Should be in the form:\n\tpython listener.py" +
          " --hosts [host1, ...] --cpus [#cpus1, ...] --jobfile path-to-file.py")
    exit(0)
    
if len(host_lst) != len(cpu_lst):
    print("Bad args. Should be in the form:\n\tpython listener.py" +
          " --hosts [host1, ...] --cpus [#cpus1, ...] --jobfile path-to-file.py")
    exit(0)

#save logfiles
logging.basicConfig(filename="log.out")
x = threading.Thread(target=listener)
x.daemon = True
print("Starting daemon")
x.start()

cmd = formHorovodCommand(host_lst, cpu_lst, jobfile, 1, epochs)
startHorovod(cmd)

#idle until the horovod process is complete or the listener thread
#encounters an error and exits
while((horovod_proc.poll() == None or restarting) and x.isAlive()):
    time.sleep(2)

#if we exited as a result of the listener thread quitting,
#shut down the current running instance of horovod as well
if horovod_proc.poll() == None:
    killHorovod()
    
print("shutting down")
