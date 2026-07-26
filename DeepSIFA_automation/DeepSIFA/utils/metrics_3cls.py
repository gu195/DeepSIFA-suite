from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
import torch

def confusion_matrix(output, target):#attention
    y_pred = output.argmax(1)
    y_true = target.argmax(1)
    mat = metrics.confusion_matrix(y_true, y_pred)
    # Add the data in the third column to the second column
    mat[:2, 1] += mat[:2, 2]
    mat = mat[:2,:2]
    return mat

def ACC(output, target):
    y_pred = output.argmax(1)
    y_true = target.argmax(1)
    mat = metrics.confusion_matrix(y_true, y_pred)
    # Add the data in the third column to the second column
    mat[:2, 1] += mat[:2, 2]
    # Calculate the total number of samples in the first two categories
    total_samples = np.sum(mat[:2, :2])
    # Calculate the original values of the first two types of diagonals
    true_positives = np.sum(np.diag(mat[:2, :2]))
    accuracy = true_positives / total_samples
    return accuracy

def Cohen_Kappa(output, target):
    y_pred = output.argmax(1)
    y_true = target.argmax(1)
    return metrics.cohen_kappa_score(y_true, y_pred)

def F1_score(output, target):
    y_pred = output.argmax(1)
    y_true = target.argmax(1)
    mat = metrics.confusion_matrix(y_true, y_pred)
    # Add the data in the third column to the second column
    mat[:2, 1] += mat[:2, 2]
    true_positives = np.diag(mat[:2, :2])[1]
    true_negatives = np.diag(mat[:2, :2])[0]
    false_positives = np.sum(mat[:2, 1]) - true_positives
    false_negatives = np.sum(mat[:2, 0]) - true_negatives

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Calculate F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    return f1_scores

def Recall_F(output, target):
    y_pred = output.argmax(1)
    y_true = target.argmax(1)
    return metrics.recall_score(y_true, y_pred, zero_division=1, average=None)

def Recall_F_F0(output, target):
    recall = []
    con_mat = confusion_matrix(output, target)

    number = np.sum(con_mat[1,:])
    tp = con_mat[1][1]
    recall = tp / number
    return recall  # returns the specificity of the first category (index 0)

def Precision(output, target):
    y_pred = output.argmax(1)
    y_true = target.argmax(1)
    mat = metrics.confusion_matrix(y_true, y_pred)
    # Add the data in the third column to the second column
    mat[:2, 1] += mat[:2, 2]
    true_positives = np.diag(mat[:2, :2])[1]
    false_positives = np.sum(mat[:2, 1]) - true_positives

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    return precision

def PPV(output, target):
    y_pred = output.argmax(1)
    y_true = target.argmax(1)
    return metrics.precision_score(y_true, y_pred, average='macro')

def NPV(output, target):
    conf_matrix = confusion_matrix(output, target)
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]  
    NPV = TN / (TN + FN)
    return NPV

def cls_report(output, target):
    y_pred = output.argmax(1)
    y_true = target.argmax(1)
    return metrics.classification_report(y_true, y_pred, digits=4)



# def spe(output, target):
#     spe = []
#     con_mat = confusion_matrix(output, target)
#     for i in range(2):
#         number = np.sum(con_mat[:,:])
#         tp = con_mat[i][i]
#         fn = np.sum(con_mat[i,:]) - tp
#         fp = np.sum(con_mat[:,i]) - tp
#         tn = number - tp - fn - fp
#         spe1 = tn / (tn + fp)
#         spe.append(spe1)
#     return spe[1]
def spe(output, target):
    spe = []
    con_mat = confusion_matrix(output, target)

    number = np.sum(con_mat[0,:])
    tn = con_mat[0][0]
    spe = tn / number
    return spe  # returns the specificity of the first category (index 0)



def roc(output, target, results_dir, file_name):

    y_true = target
    output = torch.tensor(output)
    probabilities = F.softmax(output, dim=1)
    values = []
    values0 = []

    for i in range(len(y_true)):
        selected_value = probabilities[i][1]
        values.append(selected_value.item())
        selected_value0 = probabilities[i][0]
        values0.append(selected_value0.item())
    y_pred = values
    y_pred0 = values0


    if len(np.unique(y_true)) == 1:
        auc = 0.5  # If there is only one category, set AUC to 0.5 or other appropriate value
    else:
        auc = metrics.roc_auc_score(y_true, y_pred)
        auc0 = metrics.roc_auc_score(y_true, y_pred0)
    
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.title(file_name+'auc{:.2f}'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot(fpr, tpr, label='AUC=%.4f' % auc)
    # file_name = 'roc_1.png'
    plt.savefig(os.path.join(results_dir, file_name+'auc{:.2f}.jpg'.format(auc)))
    print('auc={}'.format(auc))
    print('auc0={}'.format(auc0))
    return y_true, y_pred
