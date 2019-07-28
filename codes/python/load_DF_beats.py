import numpy as np
from scipy import signal
import scipy
from numpy import array
import os
from sklearn.decomposition import PCA
from sklearn import svm
import pandas as pd
from codes.python import ecg_waveform_extractor as waveform
import time as system_time
from scipy import stats
import warnings
from codes.python import post_process_features_ex as post_features

from sklearn.metrics import classification_report, confusion_matrix


def get_all_dataframe(patient_l_1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230],patient_l_2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234], ls = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'e', 'j', 'V', 'E', 'F'], ls2 = ["['N']","['L']", "['R']", "['A']", "['a']", "['J']", "['S']",  "['e']", "['j']", "['V']", "['E']", "['F']"] ):
    

    DB1_patients = pd.read_csv("database/DB1_patient_list.csv")
    #DB1_patients = DB1_patients.drop([1872])
    DB1_patients = DB1_patients[DB1_patients['2'].isin(ls)]
    DB1_patients = DB1_patients[DB1_patients['0'].isin(patient_l_1)]


    DB2_patients = pd.read_csv("database/DB2_patient_list.csv")
    #DB2_patients= DB2_patients.drop([18692, 31865])
    DB2_patients = DB2_patients[DB2_patients['2'].isin(ls)]
    DB2_patients = DB2_patients[DB2_patients['0'].isin(patient_l_2)]


    DB1 = pd.read_csv("database/DB1.csv")
    #DB1 = DB1.drop([1872])
    DB1 = DB1[DB1['class_beat'].isin(ls)]
    DB1 = DB1[DB1['patient'].isin(patient_l_1)]


    DB2 = pd.read_csv("database/DB2.csv")
    #DB2 = DB2.drop([18692, 31865])
    DB2 = DB2[DB2['class_beat'].isin(ls)]
    DB2 = DB2[DB2['patient'].isin(patient_l_2)]

    patients_ls_1 = DB1_patients.iloc[:,0]
    patients_ls_2 = DB2_patients.iloc[:,0]

    patients_ls_1_ID = DB1_patients.iloc[:,1]
    patients_ls_2_ID = DB2_patients.iloc[:,1]

    patients_ls_1_type = DB1_patients.iloc[:,2]
    patients_ls_2_type = DB2_patients.iloc[:,2]

    #patients_ls_1 = np.asarray(DB1_patients, dtype=int)
    patients_ls_1 = [int(i) for i in patients_ls_1]
    patients_ls_1 = np.asarray(patients_ls_1, dtype=int)
    patients_ls_1 = patients_ls_1.reshape(patients_ls_1.shape[0],1)
    #patients_ls_2 = np.asarray(DB2_patients, dtype=int)
    patients_ls_2 = [int(i) for i in patients_ls_2]
    patients_ls_2 = np.asarray(patients_ls_2, dtype=int)
    patients_ls_2 = patients_ls_2.reshape(patients_ls_2.shape[0],1)

    patients_ls_all = np.vstack((patients_ls_1,patients_ls_2))

    #patients_ls_all = patients_ls_all.reshape(7980,1)

    patients_ls_all = [int(i) for i in patients_ls_all]

    patients_ls_2 = [int(i) for i in patients_ls_2]
    patients_ls_1 = [int(i) for i in patients_ls_1]


    DB1_non_cli = pd.read_csv("database/DB1_non_clinic.csv")
    #DB1_non_cli = DB1_non_cli.drop([1872])
    DB1_non_cli = DB1_non_cli[DB1_non_cli['patient'].isin(patient_l_1)]


    DB1_non_cli = DB1_non_cli[DB1_non_cli['y0'].isin(ls)]


    DB2_non_cli = pd.read_csv("database/DB2_non_clinic.csv")
    #DB2_non_cli= DB2_non_cli.drop([18692, 31865])
    DB2_non_cli = DB2_non_cli[DB2_non_cli['y0'].isin(ls)]
    DB2_non_cli = DB2_non_cli[DB2_non_cli['patient'].isin(patient_l_2)]
 

    DB1_V1 = pd.read_csv("database/DB1_V1.csv")
    #DB1_V1 = DB1_V1.drop([1872])
    DB1_V1 = DB1_V1[DB1_V1['class_beat'].isin(ls)]
    DB1_V1 = DB1_V1[DB1_V1['patient'].isin(patient_l_1)]



    DB2_V1 = pd.read_csv("database/DB2_V1.csv")
    #DB2_V1 = DB2_V1.drop([18692, 31865])
    DB2_V1 = DB2_V1[DB2_V1['class_beat'].isin(ls)]
    DB2_V1 = DB2_V1[DB2_V1['patient'].isin(patient_l_2)]

                
   
    #ls.extend([ "['P']","[ '/']"," ['f']", "['u']"])
    DB1_dwt = pd.read_csv("database/DB1_DTW_MLII.csv")
    #DB1_dwt = DB1_dwt.drop([1872])
    DB1_dwt = DB1_dwt[DB1_dwt['beat_type'].isin(ls2)]
    DB1_dwt = DB1_dwt[DB1_dwt['patient'].isin(patient_l_1)]


    DB2_dwt = pd.read_csv("database/DB2_DTW_MLII.csv")
    #DB2_dwt = DB2_dwt.drop([18692, 31865])
    DB2_dwt = DB2_dwt[DB2_dwt['beat_type'].isin(ls2)]
    DB2_dwt = DB2_dwt[DB2_dwt['patient'].isin(patient_l_2)]


    DB1_dwt_V1 = pd.read_csv("database/DB1_DTW_V1.csv")
    #DB1_dwt_V1 = DB1_dwt_V1.drop([1872])
    DB1_dwt_V1 = DB1_dwt_V1[DB1_dwt_V1['beat_type'].isin(ls2)]
    DB1_dwt_V1 = DB1_dwt_V1[DB1_dwt_V1['patient'].isin(patient_l_1)]



    DB2_dwt_V1 = pd.read_csv("database/DB2_DTW_V1.csv")
    #DB2_dwt_V1 = DB2_dwt_V1.drop([18692, 31865])
    DB2_dwt_V1 = DB2_dwt_V1[DB2_dwt_V1['beat_type'].isin(ls2)]
    DB2_dwt_V1 = DB2_dwt_V1[DB2_dwt_V1['patient'].isin(patient_l_2)]

    #list(DB1.columns.values)
    variables_1 = DB1.iloc[:,0:130]
    class_beat_1 = DB1_patients.iloc[:,2]
    class_ID_1 = DB1_patients.iloc[:,1]

    np_variables_1 = np.asarray(variables_1)
    np_class_beat_1 = np.asarray(class_beat_1)
    np_class_ID_1 = np.asarray(class_ID_1)

    np_class_beat_1 = np_class_beat_1.reshape(np_class_beat_1.shape[0],1)
    np_class_ID_1 = np_class_ID_1.reshape(np_class_ID_1.shape[0],1)

    variables_2 = DB2.iloc[:,0:130]
    class_beat_2 = DB2_patients.iloc[:,2]
    class_ID_2 = DB2_patients.iloc[:,1]

    np_variables_2 = np.asarray(variables_2)
    np_class_beat_2 = np.asarray(class_beat_2)
    np_class_ID_2 = np.asarray(class_ID_2)

    np_class_beat_2 = np_class_beat_2.reshape(np_class_beat_2.shape[0],1)
    np_class_ID_2 = np_class_ID_2.reshape(np_class_ID_2.shape[0],1)

    DB_var_all = np.vstack((np_variables_1,np_variables_2))
    DB_class_all = np.vstack((np_class_ID_1,np_class_ID_2))
    DB_type_all = np.vstack((np_class_beat_1,np_class_beat_2))

    np_class_beat_1 = [str(i) for i in np_class_beat_1]
    np_class_ID_1 = [int(i) for i in np_class_ID_1]

    np_class_beat_1 = np.asarray(np_class_beat_1)
    np_class_ID_1 = np.asarray(np_class_ID_1)


    np_class_beat_2 = [str(i) for i in np_class_beat_2]
    np_class_ID_2 = [int(i) for i in np_class_ID_2]

    np_class_beat_2 = np.asarray(np_class_beat_2)
    np_class_ID_2 = np.asarray(np_class_ID_2)

    DB_type_all = [str(i) for i in DB_type_all]
    DB_class_all = [int(i) for i in DB_class_all]

    DB_type_all = np.asarray(DB_type_all)
    DB_class_all = np.asarray(DB_class_all)

    np_non_1 = np.asarray(DB1_non_cli)
    np_non_2 = np.asarray(DB2_non_cli)

    np_non_var_1 = np.asarray(DB1_non_cli.iloc[:,0:140])



    np_non_var_2 = np.asarray(DB2_non_cli.iloc[:,0:140])

    DB_var_non_all = np.vstack((np_non_var_1,np_non_var_2))

    dis_1 = DB1_dwt.iloc[:,0]


    dis_2 = DB2_dwt.iloc[:,0]

    dis_1_V1 = DB1_dwt_V1.iloc[:,0]


    dis_2_V1 = DB2_dwt_V1.iloc[:,0]

    dtw_1 = np.asarray(dis_1)
    dtw_1 = dtw_1.reshape(dtw_1.shape[0],1)

    dtw_2 = np.asarray(dis_2)
    dtw_2 = dtw_2.reshape(dtw_2.shape[0],1)

    dtw_1_V1 = np.asarray(dis_1_V1)
    dtw_1_V1 = dtw_1_V1.reshape(dtw_1_V1.shape[0],1)


    dtw_2_V1 = np.asarray(dis_2_V1)
    dtw_2_V1 = dtw_2_V1.reshape(dtw_2_V1.shape[0],1)

    v1_1 = DB1_V1.iloc[:,0:130]

    np_V1_1 = np.asarray(v1_1)


    v1_2 = DB2_V1.iloc[:,0:130]

    np_V1_2 = np.asarray(v1_2)

    DB_v_var_all = np.vstack((np_V1_1,np_V1_2))

    dtw_clinic_1 = np.hstack((dtw_1, dtw_1_V1))
    dtw_clinic_2 = np.hstack((dtw_2, dtw_2_V1))


    

    np_clinic_1 = np.hstack((np_variables_1, np_V1_1, dtw_clinic_1))
    np_clinic_2 = np.hstack((np_variables_2,np_V1_2, dtw_clinic_2))

    np_clinic_all = np.hstack((DB_var_all,DB_v_var_all))
    np_clinic_all = np.hstack((DB_var_all,DB_v_var_all))



    return  np_clinic_1, np_clinic_2,np_non_var_1, np_non_var_2, np_class_ID_1, np_class_ID_2, patients_ls_1, patients_ls_2, DB1, DB2, DB1_V1, DB2_V1, DB1_non_cli, DB2_non_cli, DB1_dwt, DB2_dwt, DB1_dwt_V1, DB2_dwt_V1


def get_all_dataframe_patient_specific(num_patient=300,patient_l_1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230],patient_l_2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234], ls = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'e', 'j', 'V', 'E', 'F'], ls2 = ["['N']","['L']", "['R']", "['A']", "['a']", "['J']", "['S']",  "['e']", "['j']", "['V']", "['E']", "['F']"] ):
    

    DB1_patients = pd.read_csv("database/DB1_patient_list.csv")
    #DB1_patients = DB1_patients.drop([1872])
    DB1_patients = DB1_patients[DB1_patients['2'].isin(ls)]
    DB1_patients = DB1_patients[DB1_patients['0'].isin(patient_l_1)]


    DB2_patients = pd.read_csv("database/DB2_patient_list.csv")
    #DB2_patients= DB2_patients.drop([18692, 31865])
    DB2_patients = DB2_patients[DB2_patients['2'].isin(ls)]
    DB2_patients = DB2_patients[DB2_patients['0'].isin(patient_l_2)]


    DB1 = pd.read_csv("database/DB1.csv")
    #DB1 = DB1.drop([1872])
    DB1 = DB1[DB1['class_beat'].isin(ls)]
    DB1 = DB1[DB1['patient'].isin(patient_l_1)]


    DB2 = pd.read_csv("database/DB2.csv")
    #DB2 = DB2.drop([18692, 31865])
    DB2 = DB2[DB2['class_beat'].isin(ls)]
    DB2 = DB2[DB2['patient'].isin(patient_l_2)]

    patients_ls_1 = DB1_patients.iloc[:,0]
    patients_ls_2 = DB2_patients.iloc[:,0]

    patients_ls_1_ID = DB1_patients.iloc[:,1]
    patients_ls_2_ID = DB2_patients.iloc[:,1]

    patients_ls_1_type = DB1_patients.iloc[:,2]
    patients_ls_2_type = DB2_patients.iloc[:,2]

    #patients_ls_1 = np.asarray(DB1_patients, dtype=int)
    patients_ls_1 = [int(i) for i in patients_ls_1]
    patients_ls_1 = np.asarray(patients_ls_1, dtype=int)
    patients_ls_1 = patients_ls_1.reshape(patients_ls_1.shape[0],1)
    #patients_ls_2 = np.asarray(DB2_patients, dtype=int)
    patients_ls_2 = [int(i) for i in patients_ls_2]
    patients_ls_2 = np.asarray(patients_ls_2, dtype=int)
    patients_ls_2 = patients_ls_2.reshape(patients_ls_2.shape[0],1)

    patients_ls_all = np.vstack((patients_ls_1,patients_ls_2))

    #patients_ls_all = patients_ls_all.reshape(7980,1)

    patients_ls_all = [int(i) for i in patients_ls_all]

    patients_ls_2 = [int(i) for i in patients_ls_2]
    patients_ls_1 = [int(i) for i in patients_ls_1]


    DB1_non_cli = pd.read_csv("database/DB1_non_clinic.csv")
    #DB1_non_cli = DB1_non_cli.drop([1872])
    DB1_non_cli = DB1_non_cli[DB1_non_cli['patient'].isin(patient_l_1)]


    DB1_non_cli = DB1_non_cli[DB1_non_cli['y0'].isin(ls)]


    DB2_non_cli = pd.read_csv("database/DB2_non_clinic.csv")
    #DB2_non_cli= DB2_non_cli.drop([18692, 31865])
    DB2_non_cli = DB2_non_cli[DB2_non_cli['y0'].isin(ls)]
    DB2_non_cli = DB2_non_cli[DB2_non_cli['patient'].isin(patient_l_2)]
 

    DB1_V1 = pd.read_csv("database/DB1_V1.csv")
    #DB1_V1 = DB1_V1.drop([1872])
    DB1_V1 = DB1_V1[DB1_V1['class_beat'].isin(ls)]
    DB1_V1 = DB1_V1[DB1_V1['patient'].isin(patient_l_1)]



    DB2_V1 = pd.read_csv("database/DB2_V1.csv")
    #DB2_V1 = DB2_V1.drop([18692, 31865])
    DB2_V1 = DB2_V1[DB2_V1['class_beat'].isin(ls)]
    DB2_V1 = DB2_V1[DB2_V1['patient'].isin(patient_l_2)]

                
   
    #ls.extend([ "['P']","[ '/']"," ['f']", "['u']"])
    DB1_dwt = pd.read_csv("database/DB1_DTW_MLII.csv")
    #DB1_dwt = DB1_dwt.drop([1872])
    DB1_dwt = DB1_dwt[DB1_dwt['beat_type'].isin(ls2)]
    DB1_dwt = DB1_dwt[DB1_dwt['patient'].isin(patient_l_1)]


    DB2_dwt = pd.read_csv("database/DB2_DTW_MLII.csv")
    #DB2_dwt = DB2_dwt.drop([18692, 31865])
    DB2_dwt = DB2_dwt[DB2_dwt['beat_type'].isin(ls2)]
    DB2_dwt = DB2_dwt[DB2_dwt['patient'].isin(patient_l_2)]


    DB1_dwt_V1 = pd.read_csv("database/DB1_DTW_V1.csv")
    #DB1_dwt_V1 = DB1_dwt_V1.drop([1872])
    DB1_dwt_V1 = DB1_dwt_V1[DB1_dwt_V1['beat_type'].isin(ls2)]
    DB1_dwt_V1 = DB1_dwt_V1[DB1_dwt_V1['patient'].isin(patient_l_1)]



    DB2_dwt_V1 = pd.read_csv("database/DB2_DTW_V1.csv")
    #DB2_dwt_V1 = DB2_dwt_V1.drop([18692, 31865])
    DB2_dwt_V1 = DB2_dwt_V1[DB2_dwt_V1['beat_type'].isin(ls2)]
    DB2_dwt_V1 = DB2_dwt_V1[DB2_dwt_V1['patient'].isin(patient_l_2)]

    #list(DB1.columns.values)
    variables_1 = DB1.iloc[:,0:130]
    class_beat_1 = DB1_patients.iloc[:,2]
    class_ID_1 = DB1_patients.iloc[:,1]

    np_variables_1 = np.asarray(variables_1)
    np_class_beat_1 = np.asarray(class_beat_1)
    np_class_ID_1 = np.asarray(class_ID_1)

    np_class_beat_1 = np_class_beat_1.reshape(np_class_beat_1.shape[0],1)
    np_class_ID_1 = np_class_ID_1.reshape(np_class_ID_1.shape[0],1)

    variables_2 = DB2.iloc[:,0:130]
    class_beat_2 = DB2_patients.iloc[:,2]
    class_ID_2 = DB2_patients.iloc[:,1]

    np_variables_2 = np.asarray(variables_2)
    np_class_beat_2 = np.asarray(class_beat_2)
    np_class_ID_2 = np.asarray(class_ID_2)

    np_class_beat_2 = np_class_beat_2.reshape(np_class_beat_2.shape[0],1)
    np_class_ID_2 = np_class_ID_2.reshape(np_class_ID_2.shape[0],1)

    DB_var_all = np.vstack((np_variables_1,np_variables_2))
    DB_class_all = np.vstack((np_class_ID_1,np_class_ID_2))
    DB_type_all = np.vstack((np_class_beat_1,np_class_beat_2))

    np_class_beat_1 = [str(i) for i in np_class_beat_1]
    np_class_ID_1 = [int(i) for i in np_class_ID_1]

    np_class_beat_1 = np.asarray(np_class_beat_1)
    np_class_ID_1 = np.asarray(np_class_ID_1)


    np_class_beat_2 = [str(i) for i in np_class_beat_2]
    np_class_ID_2 = [int(i) for i in np_class_ID_2]

    np_class_beat_2 = np.asarray(np_class_beat_2)
    np_class_ID_2 = np.asarray(np_class_ID_2)

    DB_type_all = [str(i) for i in DB_type_all]
    DB_class_all = [int(i) for i in DB_class_all]

    DB_type_all = np.asarray(DB_type_all)
    DB_class_all = np.asarray(DB_class_all)

    np_non_1 = np.asarray(DB1_non_cli)
    np_non_2 = np.asarray(DB2_non_cli)

    np_non_var_1 = np.asarray(DB1_non_cli.iloc[:,0:140])



    np_non_var_2 = np.asarray(DB2_non_cli.iloc[:,0:140])

    DB_var_non_all = np.vstack((np_non_var_1,np_non_var_2))

    dis_1 = DB1_dwt.iloc[:,0]


    dis_2 = DB2_dwt.iloc[:,0]

    dis_1_V1 = DB1_dwt_V1.iloc[:,0]


    dis_2_V1 = DB2_dwt_V1.iloc[:,0]

    dtw_1 = np.asarray(dis_1)
    dtw_1 = dtw_1.reshape(dtw_1.shape[0],1)

    dtw_2 = np.asarray(dis_2)
    dtw_2 = dtw_2.reshape(dtw_2.shape[0],1)

    dtw_1_V1 = np.asarray(dis_1_V1)
    dtw_1_V1 = dtw_1_V1.reshape(dtw_1_V1.shape[0],1)


    dtw_2_V1 = np.asarray(dis_2_V1)
    dtw_2_V1 = dtw_2_V1.reshape(dtw_2_V1.shape[0],1)

    v1_1 = DB1_V1.iloc[:,0:130]

    np_V1_1 = np.asarray(v1_1)


    v1_2 = DB2_V1.iloc[:,0:130]

    np_V1_2 = np.asarray(v1_2)

    DB_v_var_all = np.vstack((np_V1_1,np_V1_2))

    dtw_clinic_1 = np.hstack((dtw_1, dtw_1_V1))
    dtw_clinic_2 = np.hstack((dtw_2, dtw_2_V1))


    

    np_clinic_1 = np.hstack((np_variables_1, np_V1_1, dtw_clinic_1))
    np_clinic_2 = np.hstack((np_variables_2,np_V1_2, dtw_clinic_2))

    np_clinic_all = np.hstack((DB_var_all,DB_v_var_all))
    np_clinic_all = np.hstack((DB_var_all,DB_v_var_all))


    
    
    patient_num = num_patient
    patient_num_1 = patient_num+1

    db_new_mill = pd.DataFrame(columns=DB2.columns.values)

    DB_non_new = pd.DataFrame(columns=DB2_non_cli.columns.values)

    DB_v1_new = pd.DataFrame(columns=DB2_V1.columns.values)

    DB_dtw_new = pd.DataFrame(columns=DB2_dwt.columns.values)

    DB_dtw_new_v1 = pd.DataFrame(columns=DB2_dwt_V1.columns.values)



    np_v1_2_patient = []
    np_mlii_2_patient = []
    np_non_var = []
    np_dtw_mlii = []
    np_dtw_v1 = []
    np_class_beat = []
    np_class_id = []

    for i in patient_l_2:
        dt_1 = DB2[DB2['patient'] == i]
        dt_2 = DB2_non_cli[DB2_non_cli['patient'] == i]
        dt_3 = DB2_V1[DB2_V1['patient'] == i]
        dt_4 = DB2_dwt[DB2_dwt['patient'] == i]
        dt_5 = DB2_dwt_V1[DB2_dwt_V1['patient'] == i]
        
        dt_1 = DB2[DB2['patient'] == i]


        
        for j in range(patient_num):
            np_mlii_2_patient.append(dt_1.iloc[j,:130].values)
            np_non_var.append(dt_2.iloc[j,:140].values)
            np_v1_2_patient.append(dt_3.iloc[j,:130].values)
            np_dtw_mlii.append(dt_4.iloc[j,:1].values)
            np_dtw_v1.append(dt_5.iloc[j,:1].values)
            np_class_beat.append(dt_1.iloc[j,130])
            np_class_id.append(dt_1.iloc[j,131])
            
        
        DB = DB2[DB2['patient'] == i].iloc[patient_num_1:]
        DB_non = DB2_non_cli[DB2_non_cli['patient'] == i].iloc[patient_num_1:]
        
        DB_v1 = DB2_V1[DB2_V1['patient'] == i].iloc[patient_num_1:]
        DB_dtw = DB2_dwt[DB2_dwt['patient'] == i].iloc[patient_num_1:]
        DB_dwt_V1 = DB2_dwt_V1[DB2_dwt_V1['patient'] == i].iloc[patient_num_1:]
        

        db_new_mill = db_new_mill.append(DB, ignore_index=True)
        DB_non_new = DB_non_new.append(DB_non, ignore_index=True)
        DB_v1_new = DB_v1_new.append(DB_v1, ignore_index=True)
        DB_dtw_new = DB_dtw_new.append(DB_dtw, ignore_index=True)
        
        DB_dtw_new_v1 = DB_dtw_new_v1.append(DB_dwt_V1, ignore_index=True)

    variables_2 = db_new_mill.iloc[:,:130]
    class_beat_2 = db_new_mill.iloc[:,130]
    class_ID_2 = db_new_mill.iloc[:,131]

    np_variables_2 = np.asarray(variables_2)
    np_class_beat_2 = np.asarray(class_beat_2)
    np_class_ID_2 = np.asarray(class_ID_2)

    np_class_beat_2 = np_class_beat_2.reshape(np_class_beat_2.shape[0],1)
    np_class_ID_2 = np_class_ID_2.reshape(np_class_ID_2.shape[0],1)



    np_class_beat_2 = [str(i) for i in np_class_beat_2]
    np_class_ID_2 = [int(i) for i in np_class_ID_2]

    np_class_beat_2 = np.asarray(np_class_beat_2)
    np_class_ID_2 = np.asarray(np_class_ID_2)

    np_non_2 = np.asarray(DB_non_new)

    np_non_var_2 = np.asarray(DB_non_new.iloc[:,0:140])


    dis_2 = DB_dtw_new.iloc[:,0]


    dis_2_V1 = DB_dtw_new_v1.iloc[:,0]


    dtw_2 = np.asarray(dis_2)
    dtw_2 = dtw_2.reshape(dtw_2.shape[0],1)

    dtw_2_V1 = np.asarray(dis_2_V1)
    dtw_2_V1 = dtw_2_V1.reshape(dtw_2_V1.shape[0],1)



    v1_2 = DB_v1_new.iloc[:,0:130]

    np_V1_2 = np.asarray(v1_2)



    dtw_clinic_2 = np.hstack((dtw_2, dtw_2_V1))

    norm_dtw_clinic_2 = post_features.normalised_values_multiples(dtw_clinic_2)



    np_clinic_2 = np.hstack((np_variables_2,np_V1_2, dtw_clinic_2))

    np_class_beat_2 = np_class_beat_2.reshape(np_class_beat_2.shape[0],1)
    np_class_ID_2 = np_class_ID_2.reshape(np_class_ID_2.shape[0],1)

    np_v1_2_patient = np.asarray(np_v1_2_patient) 
    np_mlii_2_patient = np.asarray(np_mlii_2_patient)
    np_dtw_mlii = np.asarray(np_dtw_mlii)
    np_dtw_v1 = np.asarray(np_dtw_v1)
    np_non_var = np.asarray(np_non_var)
    np_class_beat = np.asarray(np_class_beat)
    np_class_id = np.asarray(np_class_id)


    np_class_beat_1 = np_class_beat_1.reshape(np_class_beat_1.shape[0],1)
    np_class_ID_1 = np_class_ID_1.reshape(np_class_ID_1.shape[0],1)

    np_class_beat = np_class_beat.reshape(np_class_beat.shape[0],1)
    np_class_id = np_class_id.reshape(np_class_id.shape[0],1)


    np_dtw_2_m = np.hstack((np_dtw_mlii, np_dtw_v1))
    np_clinic_2_m = np.hstack((np_mlii_2_patient,np_v1_2_patient , np_dtw_2_m))
    np_clinic_1 = np.vstack((np_clinic_1, np_clinic_2_m))
    dtw_clinic_1 = np.vstack((dtw_clinic_1, np_dtw_2_m))
    np_non_var_1 = np.vstack((np_non_var_1, np_non_var))
    np_class_beat_1 = np.vstack((np_class_beat_1, np_class_beat))
    np_class_ID_1 = np.vstack((np_class_ID_1, np_class_id))


    return  np_clinic_1, np_clinic_2,np_non_var_1, np_non_var_2, np_class_ID_1, np_class_ID_2



