import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw



def average(numbers):
    return float(sum(numbers)) / len(numbers)   

def cal_mean_beat(all_beats):
    averages = []
    for i in range(0, len(all_beats[0])):
        averages.append(average(all_beats[:,i]))
    return averages


def dwt(mean_beat, beat):
    distance, path = fastdtw(mean_beat, beat, dist=euclidean)
    return distance

def dwt_list(mean_beat, all_beats):
    distances = []
    i = 0
    for beat in all_beats:
        distance = dwt(mean_beat,beat)
        distances.append(distance)
        
        print("processing row "+ str(i))
        i = i+1

    return distances

 

def average(numbers):
    #print("averaging")
    return float(sum(numbers)) / len(numbers)   

def normalised_values(values, averages):
    normalised = [float(i)/averages for i in values]
    print("done norm") 
    return normalised

def normalised_values_multiples(data):
    rows = data.shape[0]
    columns = data.shape[1]

    x = np.zeros((rows,columns),dtype=object)
    #y = np.zeros((rows,1), dtype=object)
    for i in range(0,columns):
        averages = average(data[:,i])
        norm = normalised_values(data[:,i], averages)
        x[0:rows,i] = norm
        print("row "+ str(i) +" done")

    return x