# Compute the wavelet descriptor for a beat
"""
from features_ECG.py
    
VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

import numpy as np
from scipy.signal import medfilt
import scipy.stats
import pywt
import operator
from numpy.polynomial.hermite import hermfit




def compute_wavelet_descriptor(beat, family, level):
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]

def compute_wavelet_patient(patient, family, level):
    coeffs_MLII = []
    coeffs_V1 = []
    for i in range(0,len(patient.segmented_R_pos)):
        coeffs_MLII.append(compute_wavelet_descriptor(patient.segmented_beat_1[i], family, level))
        coeffs_V1.append(compute_wavelet_descriptor(patient.segmented_beat_2[i], family, level))
    return coeffs_MLII, coeffs_V1




# Compute the HOS descriptor for a beat
# Skewness (3 cumulant) and kurtosis (4 cumulant)
def compute_hos_descriptor(beat, n_intervals, lag):
    hos_b = np.zeros(( (n_intervals-1) * 2))
    for i in range(0, n_intervals-1):
        pose = (lag * (i+1))
        interval = beat[int((pose -(lag/2) )):int((pose + (lag/2)))]
        
        # Skewness  
        hos_b[i] = scipy.stats.skew(interval, 0, True)

        if np.isnan(hos_b[i]):
            hos_b[i] = 0.0
            
        # Kurtosis
        hos_b[(n_intervals-1) +i] = scipy.stats.kurtosis(interval, 0, False, True)
        if np.isnan(hos_b[(n_intervals-1) +i]):
            hos_b[(n_intervals-1) +i] = 0.0
    return hos_b


def compute_hos_patient(patient, n_intervals, lag):
    hos_b_MLII = []
    hos_b_V1 = []
    for i in range(0,len(patient.segmented_R_pos)):
        hos_b_MLII.append(compute_hos_descriptor(patient.segmented_beat_1[i], n_intervals, lag))
        hos_b_V1.append(compute_hos_descriptor(patient.segmented_beat_2[i], n_intervals, lag))
    return hos_b_MLII, hos_b_V1






# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.polynomials.hermite.html
# Support Vector Machine-Based Expert System for Reliable Heartbeat Recognition
# 15 hermite coefficients!
def compute_HBF(beat):

    coeffs_hbf = np.zeros(15, dtype=float)
    coeffs_HBF_3 = hermfit(range(0,len(beat)), beat, 3) # 3, 4, 5, 6?
    coeffs_HBF_4 = hermfit(range(0,len(beat)), beat, 4)
    coeffs_HBF_5 = hermfit(range(0,len(beat)), beat, 5)
    #coeffs_HBF_6 = hermfit(range(0,len(beat)), beat, 6)

    coeffs_hbf = np.concatenate((coeffs_HBF_3, coeffs_HBF_4, coeffs_HBF_5))

    return coeffs_hbf

def compute_HBF_patient(patient):
    coeffs_hbf_MLII = []
    coeffs_hbf_V1 = []
    for i in range(0,len(patient.segmented_R_pos)):
        coeffs_hbf_MLII.append(compute_HBF(patient.segmented_beat_1[i]))
        coeffs_hbf_V1.append(compute_HBF(patient.segmented_beat_2[i]))
    return coeffs_hbf_MLII, coeffs_hbf_V1