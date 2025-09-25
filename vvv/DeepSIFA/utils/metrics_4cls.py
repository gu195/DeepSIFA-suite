from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def ACC(output, target):
    y_pred = (torch.sigmoid(output) > 0.5).int()
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    print('y_pred:',y_pred)
    print()
    return metrics.accuracy_score(y_true, y_pred)

def Cohen_Kappa(output, target):
    y_pred = (torch.sigmoid(output) > 0.5).int()
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return metrics.cohen_kappa_score(y_true, y_pred)

def F1_score(output, target):
    y_pred = (torch.sigmoid(output) > 0.5).int()
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return metrics.f1_score(y_true, y_pred, zero_division=1,average='binary')

def Recall(output, target):
    y_pred = (torch.sigmoid(output) > 0.5).int()
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return metrics.recall_score(y_true, y_pred, zero_division=1, average='binary')

def Precision(output, target):
    y_pred = (torch.sigmoid(output) > 0.5).int()
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return metrics.precision_score(y_true, y_pred, average='binary')

def PPV(output, target):
    y_pred = (torch.sigmoid(output) > 0.5).int()
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return metrics.precision_score(y_true, y_pred, average='binary')

def NPV(output, target):
    conf_matrix = confusion_matrix(output, target)
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]  
    NPV = TN / (TN + FN)
    return NPV

def cls_report(output, target):
    y_pred = (torch.sigmoid(output) > 0.5).int()
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return metrics.classification_report(y_true, y_pred, digits=4)

def confusion_matrix(output, target):
    y_pred = (torch.sigmoid(output) > 0.5).int()
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return metrics.confusion_matrix(y_true, y_pred)

def spe(output, target):
    spe = []
    con_mat = confusion_matrix(output, target)
    for i in range(2):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    return spe[1]


def roc(output, target, results_dir, file_name):
    # y_pred = torch(output) 
    y_true = target.int()
    y_true = target.cpu().numpy()
    y_pred = output.cpu().numpy()
    # auc = metrics.roc_auc_score(y_true, y_pred)
    if len(np.unique(y_true)) == 1:
        auc = 0.5  # 如果只有一个类别，设置AUC为0.5或其他适当的值
    else:
        auc = metrics.roc_auc_score(y_true, y_pred)
    
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.title(file_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot(fpr, tpr, label='AUC=%.4f' % auc)
    # file_name = 'roc_1.png'
    plt.savefig(os.path.join(results_dir, file_name))
    return