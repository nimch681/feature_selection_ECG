import pandas as pd
import numpy as np
from codes.python import load_DF_beats as DF


def get_columns():
    #np_clinic_1_old, np_clinic_2_old,np_non_var_1_old, np_non_var_2_old, np_class_ID_1_old, np_class_ID_2_old, patients_ls_1, patients_ls_2, DB1, DB2, DB1_V1, DB2_V1, DB1_non_cli, DB2_non_cli, DB1_dwt, DB2_dwt, DB1_dwt_V1, DB2_dwt_V1 = DF.get_all_dataframe(patient_l_1=patient_ls_1,patient_l_2=patient_ls_2 , ls=ls, ls2=ls2 )

    feat_labels = ["R_duration", "R_height", "R_amp0", "R_amp1","R_amp2","R_amp3", "R_amp4", "R_amp5", "R_amp6", "R_amp7", "R_amp8", "R_amp9", "R_prominence","R_areas","Q_duration", "Q_height", "Q_amp0", "Q_amp1","Q_amp2","Q_amp3", "Q_amp4", "Q_amp5", "Q_amp6", "Q_amp7", "Q_amp8", "Q_amp9", "Q_prominence","Q_areas", "S_duration", "S_height", "S_amp0", "S_amp1","S_amp2","S_amp3", "S_amp4", "S_amp5", "S_amp6", "S_amp7", "S_amp8", "S_amp9", "S_prominence","S_areas", "P_duration", "P_height", "P_amp0", "P_amp1","P_amp2","P_amp3", "P_amp4", "P_amp5", "P_amp6", "P_amp7", "P_amp8", "P_amp9", "P_prominence","P_areas", "P_neg_duration", "P_neg_height", "P_neg_amp0", "P_neg_amp1","P_neg_amp2","P_neg_amp3", "P_neg_amp4", "P_neg_amp5", "P_neg_amp6", "P_neg_amp7", "P_neg_amp8", "P_neg_amp9", "P_neg_prominence","P_neg_areas", "T_duration", "T_height", "T_amp0", "T_amp1","T_amp2","T_amp3", "T_amp4", "T_amp5", "T_amp6", "T_amp7", "T_amp8", "T_amp9", "T_prominence","T_areas","T_neg_durations","T_neg_height", "T_neg_amp0", "T_neg_amp1", "T_neg_amp2", "T_neg_amp3", "T_neg_amp4", "T_neg_amp5", "T_neg_amp6", "T_neg_amp7", "T_neg_amp8", "T_neg_amp9","T_neg_prominence","T_neg_areas", "rr_int_pre", "rr_int_post", "rr_int_10", "rr_int_50", "rr_int_all", "QRS_int", "QRS_int_10", "QRS_int_50", "PQ_int", "PQ_int_10", "PQ_int_50", "PR_int", "PR_int_10", "PR_int_50", "ST_int", "ST_int_10", "ST_int_50", "RT_int", "RT_int_10", "RT_int_50", "PT_int", "PT_int_10", "PT_int_50", "RP","TR","neg_RQ", "neg_PR", "neg_ST", "neg_RT", "neg_PT", "P_neg_T", "neg_P_neg_T"]
    feat_labels_V = ["R_duration_V", "R_height_V", "R_amp0_V", "R_amp1_V","R_amp2_V","R_amp3_V", "R_amp4_V", "R_amp5_V", "R_amp6_V", "R_amp7_V", "R_amp8_V", "R_amp9_V", "R_prominence_V","R_areas_V","Q_duration_V", "Q_height_V", "Q_amp0_V", "Q_amp1_V","Q_amp2_V","Q_amp3_V", "Q_amp4_V", "Q_amp5_V", "Q_amp6_V", "Q_amp7_V", "Q_amp8_V", "Q_amp9_V", "Q_prominence_V","Q_areas_V", "S_duration_V", "S_height_V", "S_amp0_V", "S_amp1_V","S_amp2_V","S_amp3_V", "S_amp4_V", "S_amp5_V", "S_amp6_V", "S_amp7_V", "S_amp8_V", "S_amp9_V", "S_prominence_V","S_areas_V", "P_duration_V", "P_height_V", "P_amp0_V", "P_amp1_V","P_amp2_V","P_amp3_V", "P_amp4_V", "P_amp5_V", "P_amp6_V", "P_amp7_V", "P_amp8_V", "P_amp9_V", "P_prominence_V","P_areas_V", "P_neg_duration_V", "P_neg_height_V", "P_neg_amp0_V", "P_neg_amp1_V","P_neg_amp2_V","P_neg_amp3_V", "P_neg_amp4_V", "P_neg_amp5_V", "P_neg_amp6_V", "P_neg_amp7_V", "P_neg_amp8_V", "P_neg_amp9_V", "P_neg_prominence_V","P_neg_areas_V", "T_duration_V", "T_height_V", "T_amp0_V", "T_amp1_V","T_amp2_V","T_amp3_V", "T_amp4_V", "T_amp5_V", "T_amp6_V", "T_amp7_V", "T_amp8_V", "T_amp9_V", "T_prominence_V","T_areas_V","T_neg_durations_V","T_neg_height_V", "T_neg_amp0_V", "T_neg_amp1_V", "T_neg_amp2_V", "T_neg_amp3_V", "T_neg_amp4_V", "T_neg_amp5_V", "T_neg_amp6_V", "T_neg_amp7_V", "T_neg_amp8_V", "T_neg_amp9_V","T_neg_prominence_V","T_neg_areas_V", "rr_int_pre_V", "rr_int_post_V", "rr_int_10_V", "rr_int_50_V", "rr_int_all_V", "QRS_int_V", "QRS_int_10_V", "QRS_int_50_V", "PQ_int_V", "PQ_int_10_V", "PQ_int_50_V", "PR_int_V", "PR_int_10_V", "PR_int_50_V", "ST_int_V", "ST_int_10_V", "ST_int_50_V", "RT_int_V", "RT_int_10_V", "RT_int_50_V", "PT_int_V", "PT_int_10_V", "PT_int_50_V", "RP_V","TR_V","neg_RQ_V", "neg_PR_V", "neg_ST_V", "neg_RT_V", "neg_PT_V", "P_neg_T_V", "neg_P_neg_T_V"]
    feat_labels_dtw = ["dtw1","dtw2"]
    norm_dtw = ["dtw1_v1","dtw2_v1"]
    classID = ["classID"]
    #non_clinic = list(DB1_non_cli.iloc[:,0:140].columns.values)

    feature_norm_mill = []
    for i in range(len(feat_labels)):
        feature_norm_mill.append(str(feat_labels[i]+"_n_MLII"))

    feature_norm_V1 = []
    for i in range(len(feat_labels)):
        feature_norm_V1.append(str(feat_labels[i]+"_n_V1"))
        

    c_ID = np.asarray(classID)
    f_M = np.asarray(feat_labels)
    f_V = np.asarray(feat_labels_V)
    f_d = np.asarray(feat_labels_dtw)
    #non_cli = np.asarray(non_clinic)
    norm_mlii = np.asarray(feature_norm_mill)
    norm_v1 = np.asarray(feature_norm_V1)
    norm_dtw = np.asarray(norm_dtw)


    features_clinic = np.hstack((f_M,f_V, f_d))
    #row=[]
    #for i in range(0,len(np_clinic_new_1)):
        #row.append(i)
        
    return  features_clinic,c_ID,f_M, f_V, f_d , norm_mlii, norm_v1 , norm_dtw 

#X_train_balanced_norm = pd.DataFrame(X_train_norm,columns=features_clinic_norm)
#X_test_norm = pd.DataFrame(X_test_norm,columns=features_clinic_norm)