import os, sys
import time
from random import *

#
#----------------------------------------------------------------------------------------------------
#
# MD: Simple script to submit jobs, waiting for a real scheduler
#
#----------------------------------------------------------------------------------------------------
#
# Set here the GPUs you want to select
# e.g. NGPU = [0, 1, 2, 5, 7] # list
NGPU = [0, 1, 2, 3, 4, 5, 7] # list

#
#--------------------------------------------------
#
def getGPUOccupancy(NGPU = [0,1,2,3,4,5,6,7]):

    print("Getting the GPUs occupancy")
    # Get GPU usage from nvidia-smi
    os.system("nvidia-smi > outputNVIDIA1.tmp")
    os.system("grep \'%\' outputNVIDIA1.tmp > outputNVIDIA2.tmp")
    os.system("awk \'{print $13}\'  outputNVIDIA2.tmp > outputNVIDIA3.tmp")
    os.system("rm outputNVIDIA1.tmp")
    os.system("rm outputNVIDIA2.tmp")

    # build a dictionary for the accupancy
    freeGPU={ 0:0, # 0 = free, 1 = used
              1:0,
              2:0,
              3:0,
              4:0,
              5:0,
              6:0,
              7:0 }
    
    with open('outputNVIDIA3.tmp') as f:  
        line = f.readline()
        gpuNumber = 0
	
        while line:
            occ=float(line[:-2]) # the last character is a new-line the one before is a %
            # print(line[:-2]) 
            if occ > 10.: 
                freeGPU[gpuNumber] = 1
            line = f.readline()
            gpuNumber +=1
            
    print(freeGPU)
    return freeGPU

#
#--------------------------------------------------
#
def printGPUOccupancy(dictGPU, dictJOBS):
    nj=0
    for i,j in dictGPU.items():        
        if (dictGPU[i] == 1):
            print("GPU #", i," BUSY ", dictJOBS[nj])
            nj+=1
        else:
            print("GPU #", i ," FREE ")
    print("---")
    return

#
#--------------------------------------------------
#
def generateGPUOccupancy(jobs, NGPU = [0,1,2,3,4,5,6,7]):
    # transform the NGPU and jobs lists into dictionaries, with all GPU free ( 0 = free, 1 = busy)
    dic={} 
    for i in range(len(NGPU)):
        dic[NGPU[i]] = 0

    # randomly set the occupacy
    for i in range(len(NGPU)):
        if (random()>0.5):
            dic [NGPU[i]] = 1
        else:
            dic [NGPU[i]] = 0
            jobs[i] = ''
    return dic

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
    return 

    
#
#--------------------------------------------------
#
# MAIN
#
#--------------------------------------------------
#

# DEBUG FLAG
#
dbg = False

# clean screen
print('\x1b[2J' )                   

# get the jobs you have to run from a file
# i.e. list of commands to submit 
jobs = getJobsFromFile("jobs.txt")
totJobs = len(jobs)
#
print("Total #jobs = ",totJobs)

listJobs=['']*len(NGPU) # numbered on the position of the GPU

# initial status of GPUs:
if dbg:
    # generate a fake occupancy of the GPUs for testing this script
    gpusOcc = generateGPUOccupancy(listJobs, NGPU)
else:
    # get the occupancy of the GPUs for real
    gpusOcc = getGPUOccupancy(NGPU)
printGPUOccupancy(gpusOcc, listJobs)
time.sleep(1)

# loop over the number of jobs and remove them from the list when the job is submitted
# (ignore if job completes correctly, no resubmission)
jobCounter = 0
while  len(jobs) > 0:
    print('\x1b[2J')                    # clean screen
    for g in range(len(NGPU)):
        selGPU = NGPU[g]
        if (gpusOcc[selGPU] == 0) and (len(jobs)>0):
            # command to submit the first job i.e. jobs[0] in the list
            listJobs[g] = prepareJobString('CUDA_VISIBLE_DEVICES='+str(selGPU)+' '+jobs[0], jobCounter)
            if not dbg:
                submitJob(listJobs[g])
            jobCounter+=1
            gpusOcc [selGPU] = 1
            del jobs[0]

    printGPUOccupancy(gpusOcc, listJobs)
    print("Remaining #jobs = ", len(jobs), "/",totJobs)

    if (len(jobs)>0): 
        print("ALL GPUs TAKEN...")
        print("wait 1 min before re-scan the occupancy----------------------------------------")
        if dbg:
            time.sleep(2) 
            # generate the occupancy of the GPUs for testing this script
            gpusOcc = generateGPUOccupancy(listJobs, NGPU)
        else:
            time.sleep(60)
            # get the occupancy of the GPUs for real
            gpusOcc = getGPUOccupancy(NGPU)

print("Done submitting jobs")

#cleanup
os.system("rm outputNVIDIA3.tmp")
