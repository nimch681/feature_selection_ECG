#!/usr/bin/env python

"""
load_database.py written by Chontira Chumsaeng
TODO adapted from _______________
"""


import matplotlib.pyplot as plt
import os
from codes.python import heartbeat_segmentation as hs
from codes.python import ECG_denoising as denoise
import numpy as np
import csv
import math
import wfdb
from wfdb import processing
from codes.python import ecg_waveform_extractor as waveform
from codes.python import rr_interval_extractor as rr_int
from codes.python import interval_extractor as interval
from codes.python import ecg_non_clinical_features_ex as non_clinic


# Show a 2D plot with the data in beat
def display_signal(beat):
    plt.plot(beat)
    plt.ylabel('Signal')
    plt.show()
        

class Patient_record:
    def __init__(self,filename, database):
        self.database = database
        self.filename = filename
        self.fields = []
        self.time = []
        self.MLII = []
        self.filtered_MLII = []
        self.V1 = []
        self.filtered_V1 = []
        self.annotations = []

        self.annotated_R_poses = []
        self.annotated_beat_class = []
        self.annotated_p_waves_pos = []
        self.annotated_t_waves_pos = []
        self.segmented_class_ID = []
        self.segmented_beat_class = []
        
        self.segmented_R_pos = [] #1
        self.R_pos_properites = None #15
        self.original_R_pos = []#1
        self.segmented_beat_time = []#300
        self.segmented_beat_index = []#300
        self.segmented_beat_1 = []#300
        self.segmented_beat_2 = []#300

        self.Q_points = []#1
        self.Q_points_properites = None#15
        self.S_points = []#1
        self.S_points_properites = None#15
        self.P_points = []#1
        self.P_points_properites = None#30
        self.P_neg_points = []#1
        self.T_points = []#1
        self.T_points_properites = None#30
        self.T_neg_points = []#1

        self.QRS_interval = None#4
        self.rr_interval = None#4 done
        self.P_Q_interval = None#4
        self.neg_P_Q_interval = []#1
        self.P_R_interval = None#4
        self.neg_P_R_interval = []#1

        self.S_T_interval = None#4
        self.neg_S_T_interval = []#1
        self.R_T_interval = None#4
        self.neg_R_T_interval = []#1

        self.P_T_interval = None#4
        self.neg_P_T_interval = []#1
        self.P_neg_T_interval = []#1
        self.neg_P_neg_T_interval = []#1

        self.wavelets_coeffs_MLII = []
        self.wavelets_coeffs_V1 = []
        self.coeffs_hbf_MLII = []
        self.coeffs_hbf_V1 = []#1
        self.hos_b_MLII = []
        self.hos_b_V1 = []#1


        #self.DTW = []


        #####137 or 138 if include DTW  
        
         

    def attribute(self):
        print("database, filename, fields, time, MLII, filtered_MLII, V1, filtered_V1, annotations, annotated_R_poses, annotated_beat_class, annotated_p_waves_pos, annotated_t_waves_pos, segmented_class_ID, segmented_beat_class,segmented_R_pos, segmented_valid_R, segmented_original_R, segmented_beat_1, segmented_beat_2 ")
    
    def get_beat_1(self):
        return self.segmented_beat_1

    def get_beat_2(self):
        return self.segmented_beat_2
      
    def get_r_pos(self):
        return self.segmented_R_pos
    
    def set_segmented_beats_r_pos(self,filtered=True,is_MLII=True,is_V1=False,winL=180,winR=180,rr_max = 5):
        signal_MLII = []
        signal_V1 = []
        segmented_beat_class = []
        segmented_class_ID=[]
        segmented_R_pos = []
        beat_index = []
        times= []
        print("Start segmenting records: "+ self.filename)
        if(filtered == True):
            filter_FIR = denoise.ECG_FIR_filter()
            if(is_MLII == True):
                signal_MLII = denoise.denoising_signal_FIR(self.MLII,filter_FIR)
                self.filtered_MLII = signal_MLII
                print("Filtered MLII records from : "+ self.filename)

            if(is_V1 == True):
                signal_V1 =  denoise.denoising_signal_FIR(self.V1,filter_FIR)
                self.filtered_V1 = signal_V1
                print("Filtered V1 records from : "+ self.filename)
            
        else:
            signal_MLII = self.MLII
            signal_V1 = self.V1
        if(is_V1 == True):
            print("start segmenting V1.")
            segmented_beat_2, times,beat_index, segmented_beat_class, segmented_class_ID, segmented_R_pos,orignal_r_pos  = hs.segment_beat_from_annotation(signal_V1, self.time, self.annotations, winL, winR, rr_max)
            self.segmented_beat_2 = segmented_beat_2
            print("Finished segmenting V1.")
        if(is_MLII == True):
            print("start segmenting MLII.")
            segmented_beat_1,times,beat_index, segmented_beat_class, segmented_class_ID, segmented_R_pos,orignal_r_pos  = hs.segment_beat_from_annotation(signal_MLII, self.time, self.annotations, winL, winR, rr_max)
            self.segmented_beat_1 = segmented_beat_1
            print("Finished segmenting MLII.")
            
        self.segmented_beat_time = times
        self.segmented_beat_index = beat_index
        self.segmented_beat_class = segmented_beat_class
        self.segmented_class_ID = segmented_class_ID
        self.segmented_R_pos = segmented_R_pos
        self.original_R_pos = orignal_r_pos

        print("Segmenting record "+ self.filename + " completes.")
    
   # def set_segmented_s_and_q(self, R_peaks, time_limit = 0.01, limit=50):
        #if(self.filtered_MLII == []):
            #self
        
    def set_Q_S_points_MLII(self,time_limit_from_r=0.1,sample_from_point=[5,5], to_area=False,to_savol=True, Order=9,window_len=41, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):
        print("Processing file: "+ self.filename)
        if(self.filtered_MLII == []):
            filter_FIR = denoise.ECG_FIR_filter()
            signal_MLII = denoise.denoising_signal_FIR(self.MLII,filter_FIR)
            self.filtered_MLII = signal_MLII
            print("Filtered MLII records from : "+ self.filename)
        
        if(self.segmented_R_pos == []):
            print("Finding R pos")
            self.segmented_beat_class, self.segmented_class, self.segmented_R_pos, self.segmented_R_pos = hs.r_peak_and_annotation(self.filtered_MLII, self.annotations,list(range(0,len(self.filtered_MLII))))
        
        self.Q_points, self.Q_points_properites, self.S_points, self.S_points_properites =waveform.q_s_peak_properties_extractor(self,time_limit_from_r,sample_from_point, to_area,to_savol, Order,window_len, left_limit,right_limit, distance, width,plateau_size)
        print("Done proecessing: "+ self.filename)

    def set_r_properties_MLII(self,sample_from_R=[5,5], to_area=False,to_savol=True, Order=9,window_len=41, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):
        print("Processing file: "+ self.filename)
        if(self.filtered_MLII == []):
            filter_FIR = denoise.ECG_FIR_filter()
            signal_MLII = denoise.denoising_signal_FIR(self.MLII,filter_FIR)
            self.filtered_MLII = signal_MLII
            print("Filtered MLII records from : "+ self.filename)
        
        if(self.segmented_R_pos == []):
            print("Finding R pos")
            self.segmented_beat_class, self.segmented_class, self.segmented_R_pos, self.segmented_R_pos = hs.r_peak_and_annotation(self.filtered_MLII, self.annotations,list(range(0,len(self.filtered_MLII))))
        
        self.R_pos_properites =waveform.r_peak_properties_extractor(self,sample_from_R, to_area,to_savol, Order,window_len, left_limit,right_limit, distance, width,plateau_size)
        print("Done proecessing: "+ self.filename)

    def set_P_T_points_MLII(self,time_limit_from_r=0.1,sample_from_point=[5,5], to_area=False,to_savol=False, Order=9,window_len=31, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):
        print("Processing file: "+ self.filename)
        if(self.filtered_MLII == []):
            filter_FIR = denoise.ECG_FIR_filter()
            signal_MLII = denoise.denoising_signal_FIR(self.MLII,filter_FIR)
            self.filtered_MLII = signal_MLII
            print("Filtered MLII records from : "+ self.filename)
        
        if(self.segmented_R_pos == []):
            print("Finding R pos")
            self.segmented_beat_class, self.segmented_class, self.segmented_R_pos, self.segmented_R_pos = hs.r_peak_and_annotation(self.filtered_MLII, self.annotations,list(range(0,len(self.filtered_MLII))))
        

        p_positives, p_negatives, p_properties, t_positives, t_negatives, t_properties=waveform.p_and_t_peak_properties_extractor(self,time_limit_from_r,sample_from_point, to_area,to_savol, Order,window_len, left_limit,right_limit, distance, width,plateau_size)
        
        self.P_points = p_positives
        self.P_points_properites = p_properties
        self.P_neg_points = p_negatives
        self.T_points = t_positives
        self.T_points_properites = t_properties
        self.T_neg_points = t_negatives
        print("Done proecessing: "+ self.filename)
    
    def set_rr_intervals(self,ten=True, fifty=True,all_avg=True):
        print("Processing file: "+ self.filename)
        if(self.filtered_MLII == []):
            filter_FIR = denoise.ECG_FIR_filter()
            signal_MLII = denoise.denoising_signal_FIR(self.MLII,filter_FIR)
            self.filtered_MLII = signal_MLII
            print("Filtered MLII records from : "+ self.filename)
        
        if(self.segmented_R_pos == []):
            print("Finding R pos")
            self.segmented_beat_class, self.segmented_class, self.segmented_R_pos, self.segmented_R_pos = hs.r_peak_and_annotation(self.filtered_MLII, self.annotations,list(range(0,len(self.filtered_MLII))))
        
        self.rr_interval =rr_int.rr_interval_and_average(self,ten, fifty,all_avg)
        print("Done proecessing: "+ self.filename)
    
    def set_intervals_and_averages(self,ten=True, fifty=True,all_avg=False):
        print("Processing file: "+ self.filename)
        if(self.filtered_MLII == []):
            filter_FIR = denoise.ECG_FIR_filter()
            signal_MLII = denoise.denoising_signal_FIR(self.MLII,filter_FIR)
            self.filtered_MLII = signal_MLII
            print("Filtered MLII records from : "+ self.filename)
        
        if(self.segmented_R_pos == []):
            print("Finding R pos")
            self.segmented_beat_class, self.segmented_class, self.segmented_R_pos, self.segmented_R_pos = hs.r_peak_and_annotation(self.filtered_MLII, self.annotations,list(range(0,len(self.filtered_MLII))))
        
        QRS_properties, P_Q_properties, P_Q_neg, P_R_properties, P_R_neg, S_T_properties, S_T_neg,R_T_properties, R_T_neg, P_T_properties,neg_P_T, P_T_neg, neg_P_T_neg  =interval.interval_and_average(self,ten, fifty,all_avg) 
         
        self.QRS_interval = QRS_properties
        self.P_Q_interval = P_Q_properties
        self.neg_P_Q_interval = P_Q_neg
        self.P_R_interval = P_R_properties
        self.neg_P_R_interval = P_R_neg

        self.S_T_interval = S_T_properties
        self.neg_S_T_interval = S_T_neg
        self.R_T_interval = R_T_properties
        self.neg_R_T_interval = R_T_neg

        self.P_T_interval = P_T_properties
        self.neg_P_T_interval = neg_P_T
        self.P_neg_T_interval = P_T_neg
        self.neg_P_neg_T_interval = neg_P_T_neg
        print("Done proecessing: "+ self.filename)
    
    def set_wavelet_coeff(self, family, level):
        self.wavelets_coeffs_MLII, self.wavelets_coeffs_V1 = non_clinic.compute_wavelet_patient(self, family, level)
        print("Done proecessing: "+ self.filename)
    
    def set_hos_coeff(self,n_intervals, lag):
        self.hos_b_MLII, self.hos_b_V1 = non_clinic.compute_hos_patient(self, n_intervals, lag)
        print("Done proecessing: "+ self.filename)
    
    def set_HBF_coeff(self):
        self.coeffs_hbf_MLII, self.coeffs_hbf_V1 = non_clinic.compute_HBF_patient(self)
        print("Done proecessing: "+ self.filename)





            
        
 
class ecg_database:
    def __init__(self,database):
        # Instance atributes v
        self.database = database
        self.patient_records = []
        self.filenames = []
        self.MITBIH_classes = []
        self.AAMI_classes = []
        self.data_for_classification = []
        #self.beat = np.empty([]) # record, beat, lead
        #self.class_ID = []   
        #self.valid_R = []       
        #self.R_pos = []
        #self.orig_R_pos = []
    
    def set_MIT_class(self,mitbih_classes):
        self.MITBIH_classe = mitbih_classes
    
    def set_AAMI_classes(self, classes):
        self.AAMI_classes = classes
    

    def set_filtered_MLII(self,DB):
        for r in range(0,len(self.patient_records)):
            self.patient_records[r].filtered_MLII = DB.patient_records[r].filtered_V1
 
        print("Done Setting MLII")
    
    def set_r_pos(self,DB):
        for r in range(0,len(self.patient_records)):
            self.patient_records[r].segmented_R_pos = DB.patient_records[r].R_pos_properites["peaks"]
 
        print("Done Setting r_poses")
    
    def set_beats_amp(self,DB):
        for r in range(0,len(self.patient_records)):
            self.patient_records[r].segmented_beat_1 = DB.patient_records[r].segmented_beat_2
 
        print("Done Setting beat amp")

    def attribute(self):
        print(" database, patient_records, MITBIH_classes, AAMI_classes ")
    
    def segment_beats(self,filtered=True,is_MLII=True,is_V1=False,winL=180,winR=180,rr_max = 5):

        
        for record in self.patient_records:
            record.set_segmented_beats_r_pos(filtered,is_MLII,is_V1,winL,winR,rr_max)
 
        print("Segmenting beats complete")

    def set_R_properties(self,sample_from_R=[5,5], to_area=False,to_savol=True, Order=9,window_len=41, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):

        
        for record in self.patient_records:
            record.set_r_properties_MLII(sample_from_R, to_area,to_savol, Order,window_len, left_limit,right_limit, distance, width,plateau_size)
 
        print("Done Setting R properties")

    def set_Q_and_S_points(self,time_limit_from_r=0.1,sample_from_point=[5,5], to_area=False,to_savol=True, Order=9,window_len=41, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):

        
        for record in self.patient_records:
            record.set_Q_S_points_MLII(time_limit_from_r,sample_from_point, to_area,to_savol, Order,window_len, left_limit,right_limit, distance, width,plateau_size)
 
        print("Done Setting Q and S points")

    def set_P_T_points(self,time_limit_from_r=0.1,sample_from_point=[5,5], to_area=False,to_savol=False, Order=9,window_len=31, left_limit=50,right_limit=50, distance=1, width=[0,100],plateau_size=[0,100]):

        
        for record in self.patient_records:
            record.set_P_T_points_MLII(time_limit_from_r,sample_from_point, to_area,to_savol, Order,window_len, left_limit,right_limit, distance, width,plateau_size)
 
        print("Done Setting P and T properties")
    
    def set_rr_intervals(self,ten=True,fifty=True, all_avg=True):

        
        for record in self.patient_records:
            record.set_rr_intervals(ten,fifty, all_avg) 
 
        print("Done Setting RR_intervals")
    
    def set_intervals(self,ten=True,fifty=True, all_avg=True):

        
        for record in self.patient_records:
            record.set_intervals_and_averages(ten,fifty, all_avg) 
 
        print("Done Setting all intervals")
    
    def set_wavelets_coeff(self, family, level):

        
        for record in self.patient_records:
            record.set_wavelet_coeff(family,level) 
 
        print("Done Setting all intervals")
    
    def set_hoses_coeff(self, n_intervals, lag):

        
        for record in self.patient_records:
            record.set_hos_coeff(n_intervals, lag) 
 
        print("Done Setting all intervals")

    def set_hbfs_coeff(self):

        
        for record in self.patient_records:
            record.set_HBF_coeff() 
 
        print("Done Setting all intervals")


    



        

def create_ecg_database(database,patient_records):
    db = ecg_database(database)
    record_list = load_cases_from_list(database, patient_records)
    db.patient_records = record_list
    db.filenames = patient_records
    return db


def load_mitdb():


    mitdblstring = wfdb.get_record_list("mitdb")
    mitdbls = [int(i) for i in mitdblstring]
    mitdb = []


    for i in mitdbls:
        mitdb.append(load_patient_record("mitdb", str(i)))       
    my_db = ecg_database("mitdb")

    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
    AAMI_classes = [] 
    AAMI_classes.append(['N', 'L', 'R'])                    # N
    AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
    AAMI_classes.append(['V', 'E'])                         # VEB
    AAMI_classes.append(['F'])                              # F

   

    my_db.patient_records = mitdb
    my_db.MITBIH_classes = MITBIH_classes
    my_db.AAMI_classes = AAMI_classes
    my_db.filenames = mitdblstring
    #my_db.Annotations = annotations  
    return my_db



def load_patient_record(DB_name, record_number):
    patient_record = Patient_record(record_number, DB_name)
    pathDB = os.getcwd()+'/database/'
    filename = pathDB + DB_name +"/"+ record_number
    print(filename)
    sig, fields = wfdb.rdsamp(filename, channels=[0,1])
    filename = pathDB + DB_name + "/csv/" + record_number +".csv"
    print(filename)
    f = open(filename, "r")
    reader = csv.reader(f, delimiter=',')
    next(reader) # skip first line!
    next(reader)
    time = []
    p_waves_pos = []
    t_waves_pos =[]
    MLII_index = 0
    V1_index = 1
    if int(record_number) == 114:
        MLII_index = 1
        V1_index = 0

    #MLII = []
    #V1 = []
    #time = []
    for row in reader:
        time.append((float(row[0])))
        #MLII.append((float(row[MLII_index])))
        #V1.append((float(row[V1_index])))
    f.close

    filename = pathDB + DB_name + "/csv/" + record_number +".txt"
    print(filename)
    f = open(filename, 'rt')
    next(f) # skip first line!

    annotations = []
    for line in f:
        annotations.append(line)
    f.close

    annotated_beat_type = []
    annotated_orignal_R_poses = []

    for a in annotations:
    
        aS = a.split()
            
        annotated_orignal_R_poses.append(int(aS[1]))
        annotated_beat_type.append(str(aS[2]))

    filename = pathDB + DB_name + "/p_t_wave/" + record_number +"pt.csv"
    print(filename)
    f = open(filename, "r")
    reader = csv.reader(f, delimiter=',')
    for line in reader:
    
        if (float(line[0]) == -1):
            break    
        p_waves_pos.append(float(line[0]))

    for line in reader:
        t_waves_pos.append(float(line[0]))
    
    f.close

    patient_record.filename = record_number
    patient_record.fields = fields
    patient_record.time = time
    patient_record.annotated_p_waves_pos = p_waves_pos
    patient_record.annotated_t_waves_pos = t_waves_pos
    patient_record.MLII = sig[:,MLII_index] 
    patient_record.V1 = sig[:,V1_index] 
    patient_record.annotations = annotations
    patient_record.annotated_R_poses = annotated_orignal_R_poses
    patient_record.annotated_beat_class = annotated_beat_type
    

    return patient_record

def load_cases_from_list(database,patient_list):
    record_list = []
    for p in patient_list:
        patient = load_patient_record(database, str(p))
        record_list.append(patient)
    return record_list

def display_signal_in_seconds(patient_record,signal, time_in_second):
    sum = 0
    new_signal = []
    for t in range(0,len(signal)):
        #print(mit100.time[t+1])
        if(sum <= time_in_second):
            sum= patient_record.time[t] + patient_record.time[t+1]
            new_signal.append(signal[t])

    display_signal(new_signal)





    


    


