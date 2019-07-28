from codes.python import load_database,ECG_denoising
from codes.python import QRS_detector
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import operator
from numpy import array
import sys
import csv
import os
import matplotlib.pyplot as plt
import wfdb
from wfdb import processing, plot
from codes.python import heartbeat_segmentation as shs
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import pywt
from biosppy.signals import ecg
from sklearn import metrics
#import waipy
import operator
from codes.python import ecg_waveform_extractor as waveform
import time as system_time
from scipy import stats
import warnings
import termcolor as colored
from math import*

 

def average(numbers):
    return float(sum(numbers)) / len(numbers)



mitdb = load_database.load_mitdb()
mitdb.segment_beats()
mitdb.set_R_properties() 





def average(numbers):
    return float(sum(numbers)) / len(numbers)
def peak_properties_extractor(sig,start_point=None,end_point=None,height=None, distance=None, width = None, plateau_size=None):
    sig = sig[start_point:end_point]
    peaks,properties  = np.asarray(signal.find_peaks(sig, height=height, distance=distance,width=width,plateau_size=plateau_size))
    return peaks,properties

def point_transform_to_origin(por,point):
    point_from_origin = por + point 
    return point_from_origin

def origin_to_new_point(por,point_from_origin):
    point = point_from_origin - por
    return point

def peak_duration(time,right_edge, left_edge,point_from_origin):
    right_edge = point_transform_to_origin(point_from_origin,right_edge)
    left_edge = point_transform_to_origin(point_from_origin,left_edge)
    
    return float(time[right_edge]-time[left_edge])

def sub_signal_interval(time, start_point, end_point,point_from_origin):
    start_point = point_transform_to_origin(point_from_origin,start_point)
    end_point = point_transform_to_origin(point_from_origin,end_point)
    
    return float(time[end_point]-time[start_point])

def peak_height(signal, peak, prominence,point_from_origin):
    peak = point_transform_to_origin(point_from_origin,peak)
    height = signal[peak]-(signal[peak] - prominence)
    return height

def area_under_curve(signal,time,samples,point_from_origin):
    samples = [point_transform_to_origin(i,point_from_origin) for i in samples]
    time = np.asarray(time)
    amplitude = np.asarray(signal)
    area = metrics.auc(time[samples],amplitude[samples])
    return area

def amplitude(signal,samples,point_from_origin):
    samples = [point_transform_to_origin(i,point_from_origin) for i in samples]
    signal = np.asarray(signal)
    amplitudes = signal[samples]
    return amplitudes

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

def p_and_t_peak_properties_extractor(patient,time_limit_from_r=0.1,sample_from_point=[5,5], to_area=False,to_savol=False, Order=9,window_len=31, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):
    p_peaks = []
    p_heights = []
    p_durations = []
    p_areas = []
    p_onset = []
    p_offset = []
    p_amps = []
    p_promi = []
    sigs = []
    
    t_peaks = []
    t_heights = []
    t_durations = []
    t_areas = []
    t_onset = []
    t_offset = []
    t_amps = []
    t_promi = []
    
    p_positives = []
    p_negatives = []
    t_positives = []
    t_negatives = []
    
    
    time = patient.time
    count = 0
    print("Patient file: ",patient.filename, "begins")
    
    if(patient.filtered_MLII == []):
        print("Please filter the signal")
        return
    if(patient.segmented_R_pos == []):
        print("please segment the signal to find R peak")
        return
    
    
    

    r_peaks = patient.segmented_R_pos
    q_peaks = find_Q_point(patient.filtered_MLII,patient.time, r_peaks, time_limit = 0.01,limit=50)
    s_peaks = find_S_point(patient.filtered_MLII,patient.time, r_peaks, time_limit = 0.01,limit=50)
    #q_peaks = patient.Q_points_properites["peaks"]
    #s_peaks = patient.S_points_properites["peaks"]
        
    first_r_sig = patient.filtered_MLII[q_peaks[0]-100:q_peaks[0]]
    last_r_sig = patient.filtered_MLII[s_peaks[len(s_peaks)-1]:s_peaks[len(s_peaks)-1]+100]
    
    pre_r_sig = first_r_sig
    start_pre_r = q_peaks[0]-100
    end_pre_r = q_peaks[0]
    post_r_sig = patient.filtered_MLII[s_peaks[0]:r_peaks[1]]
    start_post_r = s_peaks[0]
    end_post_r = r_peaks[1]
    
    for i in range(0,len(r_peaks)):
        ####pre_processing
        
        negative_pre = -pre_r_sig
        
        
        if(to_savol == True):
            pre_r_sig = savgol_filter(pre_r_sig,window_len,Order)
            negative_pre = savgol_filter(negative_pre,window_len,Order)

        peak,properties= peak_properties_extractor(pre_r_sig, distance=distance, width=width, plateau_size=plateau_size)
        neg_peak,neg_properties= peak_properties_extractor(negative_pre, distance=distance, width=width, plateau_size=plateau_size)
        #print(len(neg_peak),i,r_peaks[i], len(negative_pre))
        ########
        #######do operation to find the p wave
        
        abs_peak=[]
        abs_neg_peak = []
       
        
        point = 0
        duration = 0
        prominence = 0
        height = 0
        amp=0
        area=0
        offset=0
        onset = 0        
        point_neg=0
        duration_neg=0
        prominence_neg=0
        height_neg=0
        amp_neg=0
        area_neg=0
        offset_neg=0
        onset_neg = 0
        left=0
        right= 0
        p_pos=0
        p_neg=0
        
        #print(len(abs_peak), "hi")
        if(len(peak) == 0 ):
           
            abs_peak=list(range(start_pre_r,end_pre_r))
            if(len(abs_peak) == 0):
                p_pos = round((start_pre_r+end_pre_r)/2,0)
                
            
            else:
                left, right= sudo_k_mean(abs_peak, time, patient.filtered_MLII)
                p_pos = highest_peak(right, patient.filtered_MLII)
                
            onset = p_pos-5
            offset = p_pos + 5
            p_positives.append(p_pos)
            
        
        else:
            
            abs_peak = [point_transform_to_origin(p, start_pre_r) for p in peak]
            left, right= sudo_k_mean(abs_peak, time, patient.filtered_MLII)
            p_pos = highest_peak(right, patient.filtered_MLII)
            p_positives.append(p_pos)
            index_pos = find_index(abs_peak, p_pos)
            p_peak = peak[index_pos]
            
            point, duration, prominence, height, amp, area, offset, onset=find_values_in_properties(patient,patient.filtered_MLII ,p_peak, properties, index_pos, sample_from_point, start_pre_r,to_area)
            

        
        if(len(neg_peak) == 0 ):
            
            
            abs_neg_peak=list(range(start_pre_r,end_pre_r))
            if(len(abs_neg_peak) == 0):
                p_neg = round((start_pre_r+end_pre_r)/2,0)
                
            
            else:
                neg_left, neg_right = sudo_k_mean(abs_neg_peak, time, -patient.filtered_MLII)
                p_neg = highest_peak(neg_right, patient.filtered_MLII)

            onset_neg = p_neg-5
            offset_neg = p_neg + 5
            p_negatives.append(p_neg)
        
        else:
            abs_neg_peak = [point_transform_to_origin(p, start_pre_r) for p in neg_peak]
            neg_left, neg_right = sudo_k_mean(abs_neg_peak, time, -patient.filtered_MLII)
            
            p_neg = highest_peak(neg_right,-patient.filtered_MLII)
            p_negatives.append(p_neg)
            index_neg = find_index(abs_neg_peak, p_neg)
            p_neg_peak = neg_peak[index_neg]
            point_neg, duration_neg, prominence_neg, height_neg, amp_neg, area_neg, offset_neg, onset_neg=find_values_in_properties(patient,-patient.filtered_MLII ,p_neg_peak, neg_properties, index_neg, sample_from_point, start_pre_r,to_area)

        
        
        ######Turn to normal peak to find the other properties      
            
            
        p_peaks.append((p_positives,p_negatives))
        p_heights.append((height,height_neg))
        p_durations.append((duration,duration_neg))
        p_areas.append((area,area_neg))
        p_onset.append((onset,onset_neg))
       
        p_offset.append((offset,offset_neg))
        p_amps.append((amp, amp_neg))
        p_promi.append((prominence,prominence_neg))
        
        ##################################################
        negative_post = -post_r_sig
        
        if(to_savol == True):
            post_r_sig = savgol_filter(post_r_sig,window_len,Order)
            negative_post = savgol_filter(negative_post,window_len,Order)

        peak,properties= peak_properties_extractor(post_r_sig, distance=distance, width=width, plateau_size=plateau_size)
        neg_peak,neg_properties= peak_properties_extractor(negative_post, distance=distance, width=width, plateau_size=plateau_size)
        
        ########
        #######do operation to find the t wave
        
        abs_peak=[]
        abs_neg_peak = []
       
        
        point = 0
        duration = 0
        prominence = 0
        height = 0
        amp=0
        area=0
        offset=0
        onset = 0        
        point_neg=0
        duration_neg=0
        prominence_neg=0
        height_neg=0
        amp_neg=0
        area_neg=0
        offset_neg=0
        onset_neg = 0
        left=0
        right= 0
        p_pos=0
        p_neg=0
        
        
        #print(len(abs_peak), "hi")
        if(len(peak) == 0 ):
            
            abs_peak=list(range(start_post_r,end_post_r))
            
            if(len(abs_peak) == 0):
                t_pos = round((start_post_r+end_post_r)/2,0)
            
            else:
                
                left, right= sudo_k_mean(abs_peak, time, patient.filtered_MLII)
                t_pos = highest_peak(left, patient.filtered_MLII)
                
           
            onset = t_pos-5
            offset = t_pos + 5
            
            t_positives.append(t_pos)
            
        
        else:
            abs_peak = [point_transform_to_origin(p, start_post_r) for p in peak]
            left, right= sudo_k_mean(abs_peak, time, patient.filtered_MLII)
            t_pos = highest_peak(left, patient.filtered_MLII)
            
            t_positives.append(t_pos)
            index_pos = find_index(abs_peak, t_pos)
            t_peak = peak[index_pos]
           
            point, duration, prominence, height, amp, area, offset, onset=find_values_in_properties(patient,patient.filtered_MLII ,t_peak, properties, index_pos, sample_from_point, start_post_r,to_area)


        
        if(len(neg_peak) == 0 ):
            
            abs_neg_peak=list(range(start_post_r,end_post_r))
           
            if(len(abs_neg_peak) == 0):
                
                t_neg = round((start_post_r+end_post_r)/2,0)
            else:
                neg_left, neg_right= sudo_k_mean(abs_neg_peak, time, -patient.filtered_MLII)
                t_neg = highest_peak(neg_left, patient.filtered_MLII)
            
            onset_neg = t_neg-5
            offset_neg = t_neg + 5
            t_negatives.append(t_neg)
        
        else:
            abs_neg_peak = [point_transform_to_origin(p, start_post_r) for p in neg_peak]
            neg_left, neg_right = sudo_k_mean(abs_neg_peak, time, -patient.filtered_MLII)
            t_neg = highest_peak(neg_left,-patient.filtered_MLII)
            t_negatives.append(t_neg)
            index_neg = find_index(abs_neg_peak, t_neg)
            t_neg_peak = neg_peak[index_neg]
            point_neg, duration_neg, prominence_neg, height_neg, amp_neg, area_neg, offset_neg, onset_neg=find_values_in_properties(patient,-patient.filtered_MLII ,t_neg_peak, neg_properties, index_neg, sample_from_point, start_post_r,to_area)  
        

        t_peaks.append((t_positives,t_negatives))
        t_heights.append((height,height_neg))
        t_durations.append((duration,duration_neg))
        t_areas.append((area,area_neg))
        t_onset.append((onset,onset_neg))
        t_offset.append((offset,offset_neg))
        t_amps.append((amp, amp_neg))
        t_promi.append((prominence,prominence_neg))
        
        ########next wave _________________________________________
        
        if(i == len(r_peaks)-1):
            break
           
        pre_r_sig = patient.filtered_MLII[s_peaks[i]:q_peaks[i+1]]
        
       # print("before next ",patient.filtered_MLII[s_peaks[i]:q_peaks[i-1]])
        start_pre_r = s_peaks[i]
        end_pre_r = q_peaks[i+1]
        if(i == len(r_peaks)-2):
            post_r_sig = last_r_sig
            start_post_r = s_peaks[len(s_peaks)-1]
            end_post_r = s_peaks[len(s_peaks)-1]+100
            
        else:
            post_r_sig = patient.filtered_MLII[s_peaks[i+1]:q_peaks[i+2]]
            start_post_r = s_peaks[i+1]
            end_post_r = q_peaks[i+2]
            
    
    p_properties = {
        "peaks" : p_peaks,
        "durations" : p_durations,
        "prominences" : p_promi,
        "height" : p_heights,
        "amplitudes" : p_amps,
        "areas" : p_areas,
        "onset" : p_onset,
        "offset" : p_offset
    }
    
    
    t_properties = {
        "peaks" : t_peaks,
        "durations" : t_durations,
        "prominences" : t_promi,
        "height" : t_heights,
        "amplitudes" : t_amps,
        "areas" : t_areas,
        "onset" : t_onset,
        "offset" : t_offset
    }
   
        
    return p_positives, p_negatives, p_properties, t_positives, t_negatives, t_properties

def find_index(ls,value):
    index = np.where(ls==value)
        
    index = int(index[0])
    return index


def find_values_in_properties(patient,signal ,peak, properties, index, sample_from_point, start_point,to_area):

    point = point_transform_to_origin(peak,start_point)
   
    left_ips = np.asarray(properties["left_ips"])
    right_ips = np.asarray(properties["right_ips"])
    left_ips = [int(i) for i in left_ips]
    right_ips = [int(i) for i in right_ips]

        
    
    left_edge = left_ips[index]
    right_edge = right_ips[index]
        
    duration = round(peak_duration(time=patient.time,right_edge=right_edge, left_edge=left_edge,point_from_origin=start_point),3)
    prominences = np.asarray(properties["prominences"])
    prominence = prominences[index]
    height = round(peak_height(signal, point, prominence,0),3)
      
    
    amp = amplitude(patient.filtered_MLII,list(range(point-sample_from_point[0],point+sample_from_point[1])),0)
        
    area = None
        
    if(to_area==True):
        samples = list(range(left_edge,right_edge+1))
        area = round(area_under_curve(patient.filtered_MLII,patient.time,samples,start_point),3)

            
    
    offset = point_transform_to_origin(right_edge+5,start_point)
    onset = point_transform_to_origin(left_edge-5,start_point)
    
    return point, duration, prominence, height, amp, area, offset, onset


def euclidean_distance(x,y):
 
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    
def sudo_k_mean(ls, time, amp):
    first_element = ls[0]
    last_element = ls[len(ls)-1]
    
    left = []
    right = []
    
 
    left.append(first_element)
    right.append(last_element)
    for l in range(1, len(ls)-1):
        time_1 = [time[i] for i in left]
        time_2 = [time[i] for i in right]
        amp_1 = [amp[i] for i in left]
        amp_2 = [amp[i] for i in right]
       
    
        
        centroid_1_x = average(time_1)
        centroid_1_y = average(amp_1)
       # print(centroid_1, "centroid_1")
        centroid_2_x = average(time_2)
        centroid_2_y = average(amp_2)

       # print(centroid_2, "centroid_2")
       
        point = ls[l]
        time_point = time[point]
        amp_point = amp[point]
       # print(point, "point")
       # print("time", time_point)
        
        diff_1 = euclidean_distance([centroid_1_x, centroid_1_y],[time_point,amp_point])
        diff_2 = euclidean_distance([centroid_2_x, centroid_2_y],[time_point,amp_point])
        
        if(diff_1 > diff_2):
            right.append(point)
        else:
            left.append(point)
        
    return left, right 
        
        
def highest_peak(peaks, signal):
    signal = signal[peaks]
    max_signal = max(signal)
   
    index = np.where(signal==max_signal)
    
    highest = 0
    
    
    if(len(index[0]) > 1):
        first = index[0][0]
        last = index[0][len(index[0])-1]
        index = round((first+last)/2,0)
        index = int(index)
        
    else:
        index = int(index[0])
    
    highest = peaks[index]
    
    
    return highest


all_height_p = []
all_height_t = []
all_durations_p = []
all_durations_t = []

all_height_p_neg = []
all_height_t_neg = []
all_durations_p_neg = []
all_durations_t_neg = []

min_height = []
all_p_r_durations = []
all_r_t_durations = []
for patient in mitdb.patient_records:
    p_positives, p_negatives, p_properties, t_positives, t_negatives, t_properties = p_and_t_peak_properties_extractor(patient)
    p_height = np.asarray(p_properties["height"])
    t_height = np.asarray(t_properties["height"])
    
   
    
    
    p_duration = np.asarray(p_properties["durations"])
    t_duration= np.asarray(t_properties["durations"])
    print(len(p_positives),len(t_positives), len(p_negatives), len(t_negatives),len(patient.segmented_R_pos))
    
    for i in range(0,len(p_positives)):
        
        p_r_duration = sub_signal_interval(patient.time, int(p_positives[i]), int(patient.segmented_R_pos[i]),0)
        r_t_duration = sub_signal_interval(patient.time, int(patient.segmented_R_pos[i]), int(t_positives[i]),0)
        all_p_r_durations.append((patient.filename,p_positives[i],p_r_duration,i))
        all_r_t_durations.append((patient.filename, t_positives[i],r_t_duration,i))
        
        all_height_p.append((patient.filename,p_positives[i],p_height[:,0][i],i))
        all_height_t.append((patient.filename,t_positives[i],t_height[:,0][i],i))
        all_height_p_neg.append((patient.filename,p_negatives[i],p_height[:,1][i],i))
        all_height_t_neg.append((patient.filename,t_negatives[i],t_height[:,1][i],i))
        all_durations_p.append((patient.filename,p_positives[i],p_duration[:,0][i],i))
        all_durations_t.append((patient.filename,t_positives[i],t_duration[:,0][i],i))
        all_durations_p_neg.append((patient.filename,p_negatives[i],p_duration[:,1][i],i))
        all_durations_t_neg.append((patient.filename,t_negatives[i],t_duration[:,1][i],i))


p_hight = np.asarray(all_height_p)
p_height = p_hight[:,2]
x ,y  = np.unique(p_height, return_counts=True) # counting occurrence of each loan
plt.plot(x,y)
plt.show()


all_r_height = []
for patient in mitdb.patient_records:
    print(patient.filename)
    r_height = patient.R_pos_properites["height"]
    all_r_height.extend(r_height)

x ,y  = np.unique(all_r_height, return_counts=True) # counting occurrence of each loan
plt.plot(x,y)
plt.show()

