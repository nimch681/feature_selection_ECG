import numpy as np
import pickle
from sklearn import svm, metrics
from matplotlib import pyplot as plt

def get_metrics(yhat, labels):
    """
    Creates metrics to assess the model's performance.
    yhat - the predicted labels
    labels - the known labels
    @returns - [kappa, j, jkappa]
    """
    
    conf_matrix =  metrics.confusion_matrix(yhat, labels)
    kappa = get_kappa(conf_matrix, len(labels))
    j = get_j_index(conf_matrix)
    jkappa = 0.5*kappa + 0.125*j
    return [kappa, j, jkappa]

def get_class_metrics(yhat, labels):
    confusion_matrix = metrics.confusion_matrix(yhat, labels)
    Sen = float(confusion_matrix[0,0])/float(sum(confusion_matrix[0,:]))
    Ses = float(confusion_matrix[1,1])/float(sum(confusion_matrix[1,:]))
    Sev = float(confusion_matrix[2,2])/float(sum(confusion_matrix[2,:]))
    Sef = float(confusion_matrix[3,3])/float(sum(confusion_matrix[3,:]))
    Pn = float(confusion_matrix[0,0])/float(sum(confusion_matrix[:,0]))
    Ps = float(confusion_matrix[1,1])/float(sum(confusion_matrix[:,1]))
    Pv = float(confusion_matrix[2,2])/float(sum(confusion_matrix[:,2]))
    Pf = float(confusion_matrix[3,3])/float(sum(confusion_matrix[:,3]))
    return [[Sen, Pn], [Ses, Ps], [Sev, Pv], [Sef, Pf]]

def get_kappa(confusion_matrix, n_samples):
    Po = 0.0
    Pe = 0.0
    for i in range(4):
        Po += confusion_matrix[i,i]
        Pe += sum(confusion_matrix[:,i])*sum(confusion_matrix[i,:])
    Po = Po/n_samples
    Pe = Pe/n_samples**2

    return (Po - Pe)/(1 - Pe)

def get_j_index(confusion_matrix):
    Ses = float(confusion_matrix[1,1])/float(sum(confusion_matrix[1,:]))
    Sev = float(confusion_matrix[2,2])/float(sum(confusion_matrix[2,:]))
    Ps = float(confusion_matrix[1,1])/float(sum(confusion_matrix[:,1]))
    Pv = float(confusion_matrix[2,2])/float(sum(confusion_matrix[:,2]))
    return Ses + Sev + Ps + Pv