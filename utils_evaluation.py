import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from collections import Counter
import h5py
import os.path as osp
import logging
from utils_loss import Entropy

logger = logger = logging.getLogger("sfda")


######################################################
#################### main evaluation #################
######################################################
def cal_acc_main(loader, model, args, flag=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_test = True
    re = {}
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            feas, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_fea = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = 100.0 * torch.sum(torch.squeeze(predict).float() ==
                         all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    f1_ac = f1_score(all_label, torch.squeeze(
        predict).float(), average='macro')

    all_output_prob = nn.Softmax(dim=1)(all_output)
    all_fea_norm = F.normalize(all_fea)
    re["output"] = all_output.float().cpu().numpy()
    re["prob"] = all_output_prob.float().cpu().numpy()
    _, predict = torch.max(all_output, 1)
    re["pred"] = predict.float().cpu().numpy()
    re["acc"] = torch.sum(torch.squeeze(predict).float()
                          == all_label).item() / float(all_label.size()[0])
    re["label"] = all_label.cpu().numpy()
    re["fea_norm"] = all_fea_norm.cpu().numpy()

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc.mean() # This is not Accuracy!!!
    accuracy_used = aacc if args.dset == "VISDA-C" else accuracy
    aa = [f"({idx}, {str(np.round(i, 2))})" for idx, i in enumerate(acc)]
    acc = ' '.join(aa)
    
    return accuracy_used, acc, f1_ac * 100



######################################################
############ variance reduction evaluation ###########
######################################################


############ variance reduction evaluation ###########
def calculate_variance(X, y):
    n_samples, n_features = X.shape
    unique_labels = np.unique(y)
    
    # tota variance
    mu = np.mean(X, axis=0)
    total_variance = np.sum(np.linalg.norm(X - mu, axis=1) ** 2) / n_samples
    
    # inner variance
    intra_class_variance = 0
    intra_var_per_cls = {}
    for label in unique_labels:
        X_k = X[y == label]
        mu_k = np.mean(X_k, axis=0)
        intra_class_variance += np.sum(np.linalg.norm(X_k - mu_k, axis=1) ** 2)
        intra_var_per_cls[label] = np.mean(np.linalg.norm(X_k - mu_k, axis=1) ** 2)
    
    intra_class_variance /= n_samples
    
    return total_variance, intra_class_variance, intra_var_per_cls 



############ noise reduction evaluation ##############
def calculate_noise_degree(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    total_samples = np.sum(confusion_matrix)
    
    unbounded_noise_errors = {}
    unbounded_noise_num = 0.0
    all_noise_num = 0.0
    
    for i in range(num_classes):
        correct_predictions = confusion_matrix[i, i]
        all_noise_num += np.sum(confusion_matrix[i, :]) - correct_predictions
        unbounded_noise_errors[i] = []

        for j in range(num_classes):
            if i != j and confusion_matrix[i, j] > correct_predictions:
                unbounded_noise_errors[i].append((j, confusion_matrix[i, j]))
                unbounded_noise_num += confusion_matrix[i, j]
    all_noise_ratio  = all_noise_num / total_samples
    unbounded_noise_ratio = unbounded_noise_num / total_samples
    return unbounded_noise_errors, unbounded_noise_ratio, all_noise_ratio 


############ main ##############
def cal_acc_var_and_noise(loader, model, args, flag=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_test = True
    re = {}
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            feas, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_fea = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = 100.0 * torch.sum(torch.squeeze(predict).float() ==
                         all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    f1_ac = f1_score(all_label, torch.squeeze(
        predict).float(), average='macro')

    all_output_prob = nn.Softmax(dim=1)(all_output)
    all_fea_norm = F.normalize(all_fea)
    re["output"] = all_output.float().cpu().numpy()
    re["prob"] = all_output_prob.float().cpu().numpy()
    _, predict = torch.max(all_output, 1)
    re["pred"] = predict.float().cpu().numpy()
    re["acc"] = torch.sum(torch.squeeze(predict).float()
                          == all_label).item() / float(all_label.size()[0])
    re["label"] = all_label.cpu().numpy()
    re["fea_norm"] = all_fea_norm.cpu().numpy()

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc_list = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc_list.mean() # This is not Accuracy!!!
    accuracy_used = aacc if args.dset == "VISDA-C" else accuracy
    
    
    unbounded_noise_errors, unbounded_noise_ratio, all_noise_ratio = calculate_noise_degree(matrix)
    total_variance, intra_class_variance, intra_var_per_cls = calculate_variance(re["fea_norm"], re["label"])
    
    re_dataset = (accuracy_used, total_variance, intra_class_variance, unbounded_noise_ratio, all_noise_ratio)
    re_per_cls = {}
    for idx, cls_acc in enumerate(acc_list):
        re_per_cls[idx] = (cls_acc, intra_var_per_cls[idx])
    aa = [f"({idx}, {str(np.round(i, 2))}, {str(np.round(intra_var_per_cls[idx], 4))})" for idx, i in enumerate(acc_list)]
    acc = ' '.join(aa)
    
    return accuracy_used, acc, f1_ac * 100, re_dataset, re_per_cls
