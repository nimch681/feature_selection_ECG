import numpy as np 
import math
import matplotlib.pyplot as plt
from numpy import array
from codes.python import QRS_detector
import sys
import csv
import os
import operator
from numpy import array
import sys
import csv
import os
import matplotlib.pyplot as plt
import wfdb
from wfdb import processing, plot


mitdb = load_database.load_mitdb()
mit100 = mitdb.patient_records[0]
mit100.set_segmented_beats_r_pos(winL=100,winR=180)
mit100.set_r_properties_MLII()
mit100.set_Q_S_points_MLII()
mit100.set_P_T_points_MLII()
mit100.set_rr_intervals()
mit100.set_intervals_and_averages()





QRS_properties, P_Q_properties, P_Q_neg, P_R_properties, P_R_neg, S_T_properties, S_T_neg,R_T_properties, R_T_neg, P_T_properties,neg_P_T, P_T_neg, neg_P_T_neg =interval_and_average(mit100)

patient_list_1 = ["101","106","108","109","112","114","115","116","118","119","122","124","201","203","205","207","208","209","215","220","223","230"]
patient_list_2 = ["100","103","105","111","113","117","121","123","200","202","210","212","213","214","219","221","222","228","231","232","233","234"]
DB1 = load_database.create_ecg_database("mitdb",patient_list_1)
DB2 = load_database.create_ecg_database("mitdb",patient_list_2)
DB1.segment_beats()
DB2.segment_beats()
DB1.set_R_properties()
DB2.set_R_properties()
DB1.set_Q_and_S_points()
DB1.set_P_T_points()
DB1.set_rr_intervals()
DB1.set_intervals()
DB2.set_Q_and_S_points()
DB2.set_P_T_points()
DB2.set_rr_intervals()
DB2.set_intervals()


#####################################




columns = (13*7) + 5 + (3*6) +7
#rows = len(mit100.segmented_R_pos)

rows = 0

for patient in DB1.patient_records:
        rows += len(patient.segmented_beat_time)


x = np.zeros((rows,columns),dtype=object)
y = np.zeros((rows,1), dtype=object)

row_count = 0

for patient in DB1.patient_records:
        
        for i in range(0,len(patient.segmented_beat_time)):
               

                
                row = list()

                row.append(patient.R_pos_properites["durations"][i])
                
                row.append(patient.R_pos_properites["height"][i])
                row.extend(patient.R_pos_properites["amplitudes"][i])
                row.append(patient.R_pos_properites["prominences"][i])
                
                row.append(patient.Q_points_properites["durations"][i])
                row.append(patient.Q_points_properites["height"][i])
                row.extend(patient.Q_points_properites["amplitudes"][i])
                row.append(patient.Q_points_properites["prominences"][i])

                row.append(patient.S_points_properites["durations"][i])
                row.append(patient.S_points_properites["height"][i])
                row.extend(patient.S_points_properites["amplitudes"][i])
                row.append(patient.S_points_properites["prominences"][i])
                
                p_durations=np.asarray(patient.P_points_properites["durations"])
                p_height=np.asarray(patient.P_points_properites["height"])
                p_amplitudes=np.asarray(patient.P_points_properites["amplitudes"])
                p_prominence = np.asarray(patient.P_points_properites["prominences"])

                
                row.append(p_durations[i,0])
                row.append(p_height[i,0])
                row.extend(p_amplitudes[i,0])
                row.append(p_prominence[i,0])

                row.append(p_durations[i,1])
                row.append(p_height[i,1])
                row.extend(p_amplitudes[i,1])
                row.append(p_prominence[i,1])

                t_durations=np.asarray(patient.T_points_properites["durations"])
                t_height=np.asarray(patient.T_points_properites["height"])
                t_amplitudes=np.asarray(patient.T_points_properites["amplitudes"])
                t_prominence = np.asarray(patient.T_points_properites["prominences"])


                row.append(t_durations[i,0])
                row.append(t_height[i,0])
                row.extend(t_amplitudes[i,0])
                row.append(t_prominence[i,0])

                row.append(t_durations[i,1])
                row.append(t_height[i,1])
                row.extend(t_amplitudes[i,1])
                row.append(t_prominence[i,1])

                row.append(patient.rr_interval["pre"][i])
                row.append(patient.rr_interval["post"][i])
                row.append(patient.rr_interval["average_ten"][i])
                row.append(patient.rr_interval["average_fifty"][i])
                row.append(patient.rr_interval["average_all"][i])

                row.append(patient.QRS_interval["interval"][i])
                row.append(patient.QRS_interval["average_ten"][i])
                row.append(patient.QRS_interval["average_fifty"][i])

                row.append(patient.P_Q_interval["interval"][i])
                row.append(patient.P_Q_interval["average_ten"][i])
                row.append(patient.P_Q_interval["average_fifty"][i])

                row.append(patient.P_R_interval["interval"][i])
                row.append(patient.P_R_interval["average_ten"][i])
                row.append(patient.P_R_interval["average_fifty"][i])

                row.append(patient.S_T_interval["interval"][i])
                row.append(patient.S_T_interval["average_ten"][i])
                row.append(patient.S_T_interval["average_fifty"][i])

                row.append(patient.R_T_interval["interval"][i])
                row.append(patient.R_T_interval["average_ten"][i])
                row.append(patient.R_T_interval["average_fifty"][i])

                row.append(patient.P_T_interval["interval"][i])
                row.append(patient.P_T_interval["average_ten"][i])
                row.append(patient.P_T_interval["average_fifty"][i])

                row.append(patient.neg_P_Q_interval[i])
                row.append(patient.neg_P_R_interval[i])
                row.append(patient.neg_S_T_interval[i])
                row.append(patient.neg_R_T_interval[i])
                row.append(patient.neg_P_T_interval[i])
                row.append(patient.P_neg_T_interval[i])
                row.append(patient.neg_P_neg_T_interval[i])
                
                y[row_count] = patient.segmented_class_ID[i]
                x[row_count,0:columns] = row

                
                #print(DBn1[row_count])
                row_count += 1
    

    
    #time_lens.append(len(patient.segmented_beat_time[i]))
    #row.extend(mit100.segmented_beat_1[i])
    #beats_lens.append(len(patient.segmented_beat_1[i]))
        #if (len(patient.segmented_beat_1[i]) == 347):
            #print(patient.filename)
            #print(i)
            #mit207 = patient

                


patient = mit100
for i in range(0,len(patient.segmented_beat_time)):
               
    row = list()
    row.extend(patient.R_pos_properites["durations"][i])
    row.extend(patient.R_pos_properites["height"][i])
    row.extend(patient.R_pos_properites["amplitudes"][i])
    row.extend(patient.R_pos_properites["prominences"][i])

    row.extend(patient.Q_points_properites["durations"][i])
    row.extend(patient.Q_points_properites["height"][i])
    row.extend(patient.Q_points_properites["amplitudes"][i])
    row.extend(patient.Q_points_properites["prominences"][i])

    row.extend(patient.S_points_properites["durations"][i])
    row.extend(patient.S_points_properites["height"][i])
    row.extend(patient.S_points_properites["amplitudes"][i])
    row.extend(patient.S_points_properites["prominences"][i])

    p_durations=np.asarray(patient.P_points_properites["durations"])
    p_height=np.asarray(patient.P_points_properites["height"])
    p_amplitudes=np.asarray(patient.P_points_properites["amplitudes"])
    p_prominence = np.asarray(patient.P_points_properites["prominences"])


    row.extend(p_durations[i,0])
    row.extend(p_height[i,0])
    row.extend(p_amplitudes[i,0])
    row.extend(p_prominence[i,0])

    row.extend(p_durations[i,1])
    row.extend(p_height[i,1])
    row.extend(p_amplitudes[i,1])
    row.extend(p_prominence[i,1])

    t_durations=np.asarray(patient.T_points_properites["durations"])
    t_height=np.asarray(patient.T_points_properites["height"])
    t_amplitudes=np.asarray(patient.T_points_properites["amplitudes"])
    t_prominence = np.asarray(patient.T_points_properites["prominences"])


    row.extend(t_durations[i,0])
    row.extend(t_height[i,0])
    row.extend(t_amplitudes[i,0])
    row.extend(t_prominence[i,0])

    row.extend(t_durations[i,1])
    row.extend(t_height[i,1])
    row.extend(t_amplitudes[i,1])
    row.extend(t_prominence[i,1])

    row.extend(patient.rr_interval["pre"][i])
    row.extend(patient.rr_interval["post"][i])
    row.extend(patient.rr_interval["average_ten"][i])
    row.extend(patient.rr_interval["average_fifty"][i])
    row.extend(patient.rr_interval["average_all"][i])

    row.extend(patient.QRS_interval["interval"][i])
    row.extend(patient.QRS_interval["paverage_ten"][i])
    row.extend(patient.QRS_interval["average_fifty"][i])

    row.extend(patient.P_Q_interval["interval"][i])
    row.extend(patient.P_Q_interval["paverage_ten"][i])
    row.extend(patient.P_Q_interval["average_fifty"][i])

    row.extend(patient.P_R_interval["interval"][i])
    row.extend(patient.P_R_interval["paverage_ten"][i])
    row.extend(patient.P_R_interval["average_fifty"][i])

    row.extend(patient.S_T_interval["interval"][i])
    row.extend(patient.S_T_interval["paverage_ten"][i])
    row.extend(patient.S_T_interval["average_fifty"][i])

    row.extend(patient.R_T_interval["interval"][i])
    row.extend(patient.R_T_interval["paverage_ten"][i])
    row.extend(patient.R_T_interval["average_fifty"][i])

    row.extend(patient.P_T_interval["interval"][i])
    row.extend(patient.P_T_interval["paverage_ten"][i])
    row.extend(patient.P_T_interval["average_fifty"][i])

    row.extend(patient.neg_P_Q_interval[i])
    row.extend(patient.neg_P_R_interval[i])
    row.extend(patient.neg_S_T_interval[i])
    row.extend(patient.neg_R_T_interval[i])
    row.extend(patient.neg_P_T_interval[i])
    row.extend(patient.P_neg_T_interval[i])
    row.extend(patient.neg_P_neg_T_interval[i])
    

    
    #time_lens.append(len(patient.segmented_beat_time[i]))
    #row.extend(mit100.segmented_beat_1[i])
    #beats_lens.append(len(patient.segmented_beat_1[i]))
        #if (len(patient.segmented_beat_1[i]) == 347):
            #print(patient.filename)
            #print(i)
            #mit207 = patient
                
    y[i] = patient.segmented_class_ID[i]
    x[i,0:columns] = row

    #print(DBn1[row_count])
    
columns = (13*7) + 5 + (3*6) +7
#rows = len(mit100.segmented_R_pos)

rows = 0

for patient in DB2.patient_records:
        rows += len(patient.segmented_beat_time)


x2 = np.zeros((rows,columns),dtype=object)
y2 = np.zeros((rows,1), dtype=object)

row_count = 0

for patient in DB2.patient_records:
        
        for i in range(0,len(patient.segmented_beat_time)):
               
                row = list()
                row.extend(patient.R_pos_properites["durations"][i])
                row.extend(patient.R_pos_properites["height"][i])
                row.extend(patient.R_pos_properites["amplitudes"][i])
                row.extend(patient.R_pos_properites["prominences"][i])

                row.extend(patient.Q_points_properites["durations"][i])
                row.extend(patient.Q_points_properites["height"][i])
                row.extend(patient.Q_points_properites["amplitudes"][i])
                row.extend(patient.Q_points_properites["prominences"][i])

                row.extend(patient.S_points_properites["durations"][i])
                row.extend(patient.S_points_properites["height"][i])
                row.extend(patient.S_points_properites["amplitudes"][i])
                row.extend(patient.S_points_properites["prominences"][i])

                p_durations=np.asarray(patient.P_points_properites["durations"])
                p_height=np.asarray(patient.P_points_properites["height"])
                p_amplitudes=np.asarray(patient.P_points_properites["amplitudes"])
                p_prominence = np.asarray(patient.P_points_properites["prominences"])


                row.extend(p_durations[i,0])
                row.extend(p_height[i,0])
                row.extend(p_amplitudes[i,0])
                row.extend(p_prominence[i,0])

                row.extend(p_durations[i,1])
                row.extend(p_height[i,1])
                row.extend(p_amplitudes[i,1])
                row.extend(p_prominence[i,1])

                t_durations=np.asarray(patient.T_points_properites["durations"])
                t_height=np.asarray(patient.T_points_properites["height"])
                t_amplitudes=np.asarray(patient.T_points_properites["amplitudes"])
                t_prominence = np.asarray(patient.T_points_properites["prominences"])


                row.extend(t_durations[i,0])
                row.extend(t_height[i,0])
                row.extend(t_amplitudes[i,0])
                row.extend(t_prominence[i,0])

                row.extend(t_durations[i,1])
                row.extend(t_height[i,1])
                row.extend(t_amplitudes[i,1])
                row.extend(t_prominence[i,1])

                row.extend(patient.rr_interval["pre"][i])
                row.extend(patient.rr_interval["post"][i])
                row.extend(patient.rr_interval["average_ten"][i])
                row.extend(patient.rr_interval["average_fifty"][i])
                row.extend(patient.rr_interval["average_all"][i])

                row.extend(patient.QRS_interval["interval"][i])
                row.extend(patient.QRS_interval["paverage_ten"][i])
                row.extend(patient.QRS_interval["average_fifty"][i])

                row.extend(patient.P_Q_interval["interval"][i])
                row.extend(patient.P_Q_interval["paverage_ten"][i])
                row.extend(patient.P_Q_interval["average_fifty"][i])

                row.extend(patient.P_R_interval["interval"][i])
                row.extend(patient.P_R_interval["paverage_ten"][i])
                row.extend(patient.P_R_interval["average_fifty"][i])

                row.extend(patient.S_T_interval["interval"][i])
                row.extend(patient.S_T_interval["paverage_ten"][i])
                row.extend(patient.S_T_interval["average_fifty"][i])

                row.extend(patient.R_T_interval["interval"][i])
                row.extend(patient.R_T_interval["paverage_ten"][i])
                row.extend(patient.R_T_interval["average_fifty"][i])

                row.extend(patient.P_T_interval["interval"][i])
                row.extend(patient.P_T_interval["paverage_ten"][i])
                row.extend(patient.P_T_interval["average_fifty"][i])

                row.extend(patient.neg_P_Q_interval[i])
                row.extend(patient.neg_P_R_interval[i])
                row.extend(patient.neg_S_T_interval[i])
                row.extend(patient.neg_R_T_interval[i])
                row.extend(patient.neg_P_T_interval[i])
                row.extend(patient.P_neg_T_interval[i])
                row.extend(patient.neg_P_neg_T_interval[i])
    

    
    #time_lens.append(len(patient.segmented_beat_time[i]))
    #row.extend(mit100.segmented_beat_1[i])
    #beats_lens.append(len(patient.segmented_beat_1[i]))
        #if (len(patient.segmented_beat_1[i]) == 347):
            #print(patient.filename)
            #print(i)
            #mit207 = patient
                y2[row_count] = patient.segmented_class_ID[i]
                x2[row_count,0:columns] = row

                #print(DBn1[row_count])
                row_count += 1
