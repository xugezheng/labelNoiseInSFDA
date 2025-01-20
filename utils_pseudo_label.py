import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from collections import defaultdict

import logging

logger = logging.getLogger(name='sfda')


def obtain_label(loader, model, args):
    
    re_dict = get_output_dict_pure(loader, model, args)
    all_label = re_dict['org']['label']
    
    predict = get_pseudo_label(re_dict, args) # np
    
    
    # result display
    matrix = confusion_matrix(all_label, predict.astype('float'))
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    
    
    logger.info(f"PL Acc (avg per cls) = {acc:.4f}%")

    return predict


def obtain_label_pointda(loader, model_f, model_c, args):
    
    re_dict = get_output_dict_pure_pointda(loader, model_f, model_c, args)
    all_label = re_dict['org']['label']
    
    predict = get_pseudo_label(re_dict, args) # np
    # result display
    matrix = confusion_matrix(all_label, predict.astype('float'))
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100

    
    logger.info(f"PL Acc (avg per cls) = {acc:.4f}%")


    return predict


def get_output_dict_pure(loader, model, args):
    re = defaultdict(dict)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            labels = data[1]
            if args.is_data_aug or args.lln_type == "gjs":
                inputs = data[0][2].to(args.device)
            else:
                inputs = data[0].to(args.device)
            
            
            feas, outputs = model(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
            all_output_prob = nn.Softmax(dim=1)(all_output)
            _, predict = torch.max(all_output_prob, 1)
            all_fea_norm = F.normalize(all_fea)
            re['org']["output"] = all_output.float().cpu()
            re['org']["prob"] = all_output_prob.float().cpu()
            re['org']["pred"] = predict.float().cpu()
            re['org']["acc"] = torch.sum(torch.squeeze(predict).float()
                                == all_label).item() / float(all_label.size()[0])
            re['org']["label"] = all_label.cpu()
            re['org']["fea_norm"] = all_fea_norm.float().cpu()
            re['org']["fea"] = all_fea.float().cpu()
    return re


def get_output_dict_pure_pointda(loader, model_f, model_c, args):
    re = defaultdict(dict)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            labels = data[1]
            if args.is_data_aug or args.lln_type == "gjs":
                inputs = data[0][2].to(args.device)
            else:
                inputs = data[0].to(args.device)

            feas = model_f(inputs)
            outputs = model_c(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
            all_output_prob = nn.Softmax(dim=1)(all_output)
            _, predict = torch.max(all_output_prob, 1)
            all_fea_norm = F.normalize(all_fea)
            re['org']["output"] = all_output.float().cpu()
            re['org']["prob"] = all_output_prob.float().cpu()
            re['org']["pred"] = predict.float().cpu()
            re['org']["acc"] = torch.sum(torch.squeeze(predict).float()
                                == all_label).item() / float(all_label.size()[0])
            re['org']["label"] = all_label.cpu()
            re['org']["fea_norm"] = all_fea_norm.float().cpu()
            re['org']["fea"] = all_fea.float().cpu()
    return re


def get_output_dict(loader, model, args):
    re = defaultdict(dict)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            labels = data[1]
            if args.is_data_aug:
                inputs1, inputs2, inputs = (
                data[0][0].to(args.device),
                data[0][1].to(args.device),
                data[0][2].to(args.device),
            )
                feas, outputs = model(inputs)
                feas1, outputs1 = model(inputs1)
                feas2, outputs2 = model(inputs2)
                if start_test:
                    all_label = labels.float()
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_fea1 = feas1.float().cpu()
                    all_output1 = outputs1.float().cpu()
                    all_fea2 = feas2.float().cpu()
                    all_output2 = outputs2.float().cpu()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                    all_fea1 = torch.cat((all_fea1, feas1.float().cpu()), 0)
                    all_output1 = torch.cat((all_output1, outputs1.float().cpu()), 0)
                    all_fea2 = torch.cat((all_fea2, feas2.float().cpu()), 0)
                    all_output2 = torch.cat((all_output2, outputs2.float().cpu()), 0)
                all_output_prob = nn.Softmax(dim=1)(all_output)
                all_output_prob1 = nn.Softmax(dim=1)(all_output1)
                all_output_prob2 = nn.Softmax(dim=1)(all_output2)
                _, predict = torch.max(all_output_prob, 1)
                _, predict1 = torch.max(all_output_prob1, 1)
                _, predict2 = torch.max(all_output_prob2, 1)
                all_fea_norm = F.normalize(all_fea)
                all_fea_norm1 = F.normalize(all_fea1)
                all_fea_norm2 = F.normalize(all_fea2)
                
                re['aug1']["output"] = all_output1.float().cpu()
                re['aug1']["prob"] = all_output_prob1.float().cpu()
                re['aug1']["pred"] = predict1.float().cpu()
                re['aug1']["acc"] = torch.sum(torch.squeeze(predict1).float()
                                    == all_label).item() / float(all_label.size()[0])
                re['aug1']["label"] = all_label.cpu()
                re['aug1']["fea_norm"] = all_fea_norm1.cpu()
                re['aug1']["fea"] = all_fea1.cpu()
                
                re['aug2']["output"] = all_output2.float().cpu()
                re['aug2']["prob"] = all_output_prob2.float().cpu()
                re['aug2']["pred"] = predict2.float().cpu()
                re['aug2']["acc"] = torch.sum(torch.squeeze(predict2).float()
                                    == all_label).item() / float(all_label.size()[0])
                re['aug2']["label"] = all_label.cpu()
                re['aug2']["fea_norm"] = all_fea_norm2.cpu()
                re['aug2']["fea"] = all_fea2.cpu()
                
            else:
                inputs = data[0].to(args.device)
                feas, outputs = model(inputs)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                all_output_prob = nn.Softmax(dim=1)(all_output)
                _, predict = torch.max(all_output_prob, 1)
                all_fea_norm = F.normalize(all_fea)
            re['org']["output"] = all_output.float().cpu()
            re['org']["prob"] = all_output_prob.float().cpu()
            re['org']["pred"] = predict.float().cpu()
            re['org']["acc"] = torch.sum(torch.squeeze(predict).float()
                                == all_label).item() / float(all_label.size()[0])
            re['org']["label"] = all_label.cpu()
            re['org']["fea_norm"] = all_fea_norm.float().cpu()
            re['org']["fea"] = all_fea.float().cpu()
    return re
    

def get_pseudo_label(re, args):
    # SHOT pseudo-label
    all_output = re['org']['prob']
    all_label = re['org']['label']
    all_fea = re['org']['fea']
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() ==
                        all_label).item() / float(all_label.size()[0])
    
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.numpy()
    K = all_output.size(1)
    aff = all_output.numpy()


    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  # K * 256
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'SHOT PL generating | Accuracy = {:.2f}% -> {:.2f}%'.format(
        accuracy * 100, acc * 100)

    logger.info(f'{log_str}')
    args.out_file.write(log_str)
    args.out_file.flush()
    
    predict = predict.astype('int') #numpy array
    return predict
    
    
    
