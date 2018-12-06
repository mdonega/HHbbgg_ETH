import os, sys
import time
from random import *


#
#----------------------------------------------------------------------------------------------------
#
# MD: Simple script to submit jobs, waiting for a rasl scheduler
#
#----------------------------------------------------------------------------------------------------
#
#Set here the #GPU you have
#
NGPU = 8

#
#--------------------------------------------------
#
def getGPUOccupancy(NGPU = 8):

    print("Getting the GPUs occupancy")
    # Get GPU usage from nvidia-smi
    os.system("nvidia-smi > outputNVIDIA1.txt")
    os.system("grep \'%\' outputNVIDIA1.txt > outputNVIDIA2.txt")
    os.system("awk \'{print $13}\'  outputNVIDIA2.txt > outputNVIDIA3.txt")
    os.system("rm outputNVIDIA1.txt")
    os.system("rm outputNVIDIA2.txt")

    freeGPU=[0]*NGPU # 0 = free, 1 = used
    
    with open('outputNVIDIA3.txt') as f:  
        line = f.readline()
        gpuNumber = 0
	
        while line:
            occ=float(line[:-2])# the last character is a new-line the one before is a %
            # print(line[:-2]) 
            if occ > 10.: 
                freeGPU[gpuNumber] = 1
            line = f.readline()
            gpuNumber +=1
            
    # print(freeGPU)
    return freeGPU

#
#--------------------------------------------------
#
def printGPUOccupancy(vecGPU, vecJOBS):
    for i in range(len(vecGPU)):        
        if (vecGPU[i] == 1):
            print("GPU #", i ," BUSY ", vecJOBS[i])
        else:
            print("GPU #", i ," FREE ")

#
#--------------------------------------------------
#
def generateGPUOccupancy(NGPU = 8):
    vec=[0]*NGPU
    for i in range(NGPU):
        if (random()>0.5):
            vec[i]=1
        else:
            vec[0]
    return vec

#
#--------------------------------------------------
#
def getJobsFromFile(infile):

    jobs = []
    with open(infile) as f:  
        line = f.readline()
        while line:                        
            jobs.append(line[:-1])
            line = f.readline()
        
    print(jobs)        
    return jobs

#
#--------------------------------------------------
#
def prepareJobString(command, index):
    return command +  '--out-dir test' + str(index) + ' > job_counter' + str(index) + '.out 2>&1 &'


#
#--------------------------------------------------
#
def submitJob(command):
    os.system(command)        
    # print("SUBMITTING: ", command)        
    return 

    
#
#--------------------------------------------------
#
# MAIN
#
#--------------------------------------------------
#

# get the jobs you have to run from a file
# i.e. list of commands to submit 
jobs = getJobsFromFile("jobs.txt")

totJobs = len(jobs)
listJobs=['']*NGPU
jobCounter = 0

# initial status of GPUs
# sys.stderr.write("\x1b[2J\x1b[H") # clean screen
print('\x1b[2J' )                   # clean screen

# get the occupancy of the GPUs
gpusOcc = getGPUOccupancy()
printGPUOccupancy(gpusOcc, listJobs)
time.sleep(1)

# loop over the number of jobs and remove them from the list when the job is submitted
# (ignore if job completes correctly, no resubmission)
j=0
while  (j in range(len(jobs))) and (len(jobs)>0):

    # sys.stderr.write("\x1b[2J\x1b[H") # clean screen
    print('\x1b[2J')                    # clean screen

    print("Remainig #jobs = ", len(jobs), "/",totJobs)

    for g in range(NGPU):
        if (gpusOcc[g] == 0) and (len(jobs)>0):
            # listJobs[g] = 'CUDA_VISIBLE_DEVICES='+str(g)+' '+jobs[j]
            listJobs[g] = prepareJobString('CUDA_VISIBLE_DEVICES='+str(g)+' '+jobs[j], jobCounter)
            print(listJobs[g])
            submitJob(listJobs[g])
            jobCounter+=1
            gpusOcc [g] = 1
            del jobs[j]

    printGPUOccupancy(gpusOcc, listJobs)

    if (len(jobs)>0): 
        print("ALL GPUs TAKEN...")
#        printGPUOccupancy(gpusOcc, listJobs)
        print("wait 1 min before re-scan the occupancy")
        time.sleep(60)
#        gpusOcc = generateGPUOccupancy()
        gpusOcc = getGPUOccupancy()

#sys.stderr.write("\x1b[2J\x1b[H") # clean screen
#print(chr(27) + "[2J")            # clean screen
print('\x1b[2J')                   # clean screen
print("Remainig #jobs = ", len(jobs), "/",totJobs)
printGPUOccupancy(gpusOcc, listJobs)
print("Done submitting jobs")

#cleanup
os.system("rm outputNVIDIA3.txt")
