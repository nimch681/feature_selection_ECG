#!/usr/bin/env python

"""
simple_heartbeat_segmentation.py
Description:
contains code for simple heartbeat segmentation
Adapted from the code from VARPA, University of Coruna: Mondejar Guerra, Victor M.
24 Oct 2017
"""
import operator
import numpy as np
from wfdb import processing



def segment(signal,pos,winL,winR,size_RR_max):
    lst = list(signal[pos - size_RR_max : pos + size_RR_max])
    
    
    if(signal[pos] < 0):
        beat_pos = [abs(x) for x in lst]
        beat_pos = enumerate(beat_pos)
        index, value  = max(beat_pos, key=operator.itemgetter(1))
        pos = (pos - size_RR_max) + index
    
    else:
        beat_pos = enumerate(lst)
        index, value  = max(beat_pos, key=operator.itemgetter(1))
        pos = (pos - size_RR_max) + index

    beat_poses = list(range(pos - winL, pos + winR))
    beat_poses = [int(i) for i in beat_poses]
    return beat_poses,pos

def is_class_MIT(classAnttd):
    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    if classAnttd in MITBIH_classes:
        return True
    else:
        return False
        

def check_class_AAMI(classAnttd, class_AAMI):
    AAMI_classes = []
    AAMI_classes.append(['N', 'L', 'R'])                    # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
    AAMI_classes.append(['V', 'E'])                         # VEB
    AAMI_classes.append(['F'])
    for i in range(0,len(AAMI_classes)):

        if classAnttd in AAMI_classes[i]:
            class_AAMI = i      

    return class_AAMI


def segment_beat_from_annotation(signal,time,annotations, winL=180, winR=180,size_RR_max=5):
    class_ID = []
    beats = []
    times = []
    beat_index = []
    R_poses = []
    beat_class = []
    originalPoses = []
    
    
    for a in annotations:
    
        aS = a.split()
        pos = int(aS[1])
        class_AAMI = -1

        if(pos > len(signal)):
            break
            
        originalPos = int(aS[1])
        classAnttd = str(aS[2])

        if pos > size_RR_max and pos < (len(signal) - size_RR_max):
            beat_poses,pos=segment(signal,pos,winL,winR, size_RR_max)
            beat = list(signal[beat_poses[0] : beat_poses[len(beat_poses)-1]+1])
            time_beat = list(time[beat_poses[0] : beat_poses[len(beat_poses)-1]+1])

            if(pos > winL and pos < (len(signal) - winR)):
                beat_index.append(beat_poses)
                beats.append(beat)
                times.append(time_beat)
                R_poses.append(pos)
                class_AAMI = check_class_AAMI(classAnttd, class_AAMI)
                class_ID.append(class_AAMI)
                beat_class.append(classAnttd)
                originalPoses.append(originalPos)

    return beats,times,beat_index, beat_class, class_ID, R_poses, originalPoses

###have to pass in the whole index of the signal, to be start and end
def r_peak_and_annotation(signal,annotations, indexes, winL=180, winR=180,size_RR_max=5):
    class_ID = []
    
    originalPoses = []
   
    R_poses = []
    beat_class = []
    class_AAMI = -1
    
    for a in annotations:
    
        aS = a.split()
        pos = int(aS[1])

       

        if(len(signal) < len(indexes)):
            break

        if(indexes[len(indexes)-1] < pos):
            break

        if(indexes[0] > pos):
            continue
        
        originalPos = int(aS[1])
        classAnttd = str(aS[2])

        if pos > size_RR_max and pos < (len(signal) - size_RR_max):
            beat_poses,pos=segment(signal,pos,winL,winR, size_RR_max)
            beat = list(signal[beat_poses[0] : beat_poses[len(beat_poses)-1]+1])
            #time_beat = list(time[beat_poses[0] : beat_poses[len(beat_poses)-1]+1])
            pos = pos - indexes[0]
            if(pos > winL and pos < (len(indexes) - winR)):
                
                R_poses.append(pos)
                class_AAMI = check_class_AAMI(classAnttd, class_AAMI)
                class_ID.append(class_AAMI)
                beat_class.append(classAnttd)
                originalPos = originalPos-indexes[0]
                originalPoses.append(originalPos)

    return beat_class, class_ID, R_poses, originalPoses

def r_peak_detector(signal,annotations,indexes,  winL=180, winR=180,size_RR_max=5):
    
    R_poses = []
    class_AAMI = -1
    for a in annotations:
        aS = a.split()
        pos = int(aS[1])

        
        if(len(signal) < len(indexes)):
            
            break

        if(indexes[len(indexes)-1] < pos):
            
            break
        if(indexes[0] > pos): 
           
            continue

        originalPos = int(aS[1])
        classAnttd = str(aS[2])
        if pos > size_RR_max and pos < (len(signal) - size_RR_max):
            beat_poses,pos=segment(signal,pos,winL,winR, size_RR_max)
            pos = pos - indexes[0]
            if(pos > winL and pos < (len(indexes) - winR)):
                
                R_poses.append(pos)
                
    return R_poses


def annotated_r_peaks(signal,annotations,indexes,  winL=180, winR=180,size_RR_max=5):
    
    R_poses = []
    class_AAMI = -1
    for a in annotations:
        aS = a.split()
        pos = int(aS[1])
           
        if(len(signal) < len(indexes)):
            break

        if(indexes[len(indexes)-1] < pos):
            break
        if(indexes[0] > pos): 
            continue

        originalPos = int(aS[1])
        classAnttd = str(aS[2])
        if pos > size_RR_max and pos < (len(signal) - size_RR_max):
            pos = pos - indexes[0]
            if(pos > winL and pos < (len(indexes) - winR)):
                
                R_poses.append([pos,originalPos])
                
    return R_poses   

def r_peak(signal,r_peak, winL=180, winR=180,size_RR_max=5):
    
    R_poses = []
    class_AAMI = -1
    for r in r_peak:
        pos = r
        if(pos > len(signal)):
            break  
        
        if pos > size_RR_max and pos < (len(signal) - size_RR_max):
            beat_poses,pos=segment(signal,pos,winL,winR, size_RR_max)
            if(pos > winL and pos < (len(signal) - winR)):
               
                R_poses.append(pos)
                
    return R_poses


def segment_beat(signal,r_poses, winL=180, winR=180,size_RR_max=5):
    beats = []
    for pos in r_poses:
        if pos > size_RR_max and pos < (len(signal) - size_RR_max ):
            beat_poses=segment(signal,pos,winL,winR, size_RR_max)
            beat = list(signal[beat_poses[0] : beat_poses[len(beat_poses)-1]+1])
            if(pos > winL and pos < (len(signal) - winR)):
                beats.append(beat)
    return np.asrray(beats)
            


def xqrs_segment_beat(signal, fs, winL=180, winR=180, size_RR_max = 5):
    r_poses = processing.xqrs_detect(sig=signal, fs=fs)

    beat = []
    for pos in r_poses:
        if pos > size_RR_max and pos < (len(signal) - size_RR_max ):
            beat_poses=segment(signal,pos,winL,winR, size_RR_max)
            if(pos > winL and pos < (len(signal) - winR)):
                beat.append(beat_poses)
    return beat

# only works with filtered leads 
def find_Q_point(signal,time, R_peaks, time_limit = 0.01,limit=50):
    num_peak = len(R_peaks)
    Q_points = []   
    for i in range(num_peak):
        r_peak = R_peaks[i]
        point = r_peak
        if point-1 >= len(signal):
            
            break
        
        if(signal[point] >= 0 ):
            while point >= R_peaks[i] - limit and signal[point] >= signal[point - 1] or abs(time[r_peak]-time[point]) <= time_limit:             
                point -= 1
                if point >= len(signal):
                    break
        else:
            
            while point >= R_peaks[i] - limit and abs(signal[point]) >= abs(signal[point - 1]) or abs(time[r_peak]-time[point]) <= time_limit:             
                point -= 1
                if point <= len(signal):
                    break
        
        Q_points.append(point)
                        
    return np.asarray(Q_points)

# only works with filtered leads 
def find_S_point(signal,time, R_peaks, time_limit = 0.01, limit=50):
    num_peak = len(R_peaks)
    S_points = []   
    for i in range(num_peak):
        
        r_peak = R_peaks[i]
        point = r_peak
        if point+1 >= len(signal):
           
            break
        
        if(signal[point] >= 0 ):
            while point <= R_peaks[i] + limit and signal[point] >= signal[point + 1] or abs(time[point]-time[r_peak]) <= time_limit:             
                point += 1
                if point >= len(signal):
                   
                    break
        else:
            
            while  point <= R_peaks[i] + limit and abs(signal[point]) >= abs(signal[point + 1]) or abs(time[point]-time[r_peak]) <= time_limit:             
                point += 1
                if point >= len(signal):
                    break
        
        S_points.append(point)
                        
    return np.asarray(S_points)
