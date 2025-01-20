import argparse
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append("..")
# sys.path.append(PATH_TO_LABELNOISEINSFDA)
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import ast
import yaml
import pickle
from torch.utils.data import DataLoader

from loggers import TxtLogger, set_logger
from utils_lln_losses import criterion_load
from utils_loss import cross_entropy, Entropy
from utils_pseudo_label import obtain_label_pointda
## -------- pointda
import Model
from dataloader import Modelnet40_data_AUG, Shapenet_data_AUG, Scannet_data_h5_AUG, Modelnet40_data, Shapenet_data, Scannet_data_h5

def load_pointda_data(args):
    if args.is_data_aug or args.lln_type == 'gjs':
        data_func = {
            'modelnet': Modelnet40_data_AUG,
            'scannet': Scannet_data_h5_AUG,
            'shapenet': Shapenet_data_AUG,
        }
    else:
        data_func = {
            'modelnet': Modelnet40_data,
            'scannet': Scannet_data_h5,
            'shapenet': Shapenet_data,
        }
    
    dir_root = os.path.join(args.root, args.dset)
    pc_root = os.path.join(dir_root, args.target)
    BATCH_SIZE = args.batch_size
    dset_loaders = {}
    
    target_train_dataset1 = data_func[args.target](pc_input_num=1024,
                                                   status='train',
                                                   aug=True,
                                                   pc_root=pc_root)
    target_train_dataset2 = data_func[args.target](pc_input_num=1024,
                                                   status='train',
                                                   aug=True,
                                                   pc_root=pc_root)

    target_test_dataset1 = data_func[args.target](pc_input_num=1024, status='test', aug=False, pc_root= pc_root)

    num_target_train1 = len(target_train_dataset1)
    num_target_test1 = len(target_test_dataset1)

    target_train_dataloader = DataLoader(target_train_dataset1,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)
    target_pl_dataloader = DataLoader(target_train_dataset2,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=2,
                                         drop_last=False)
    target_test_dataloader = DataLoader(target_test_dataset1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=2,
                                        drop_last=False)

    print(
        'num_target_train: {:d}, num_target_test: {:d} '
        .format(num_target_train1, num_target_test1))
    print('batch_size:', BATCH_SIZE)
    args.nb_classes = args.class_num
    args.nb_samples = num_target_train1
    
    dset_loaders["target"] = target_train_dataloader
    dset_loaders["test"] = target_test_dataloader
    dset_loaders["pl"] = target_pl_dataloader
    
    return dset_loaders
    
def cal_pointda(loader, model_f, model_c, device):
    
    with torch.no_grad():
        model_f.eval()
        model_c.eval()
        correct_total = 0
        data_total = 0


        correct_total = 0
        data_total = 0
        for batch_idx, (data, label, _) in enumerate(loader):
            data = data.to(device=device)
            label = label.to(device=device).long()
            output = model_c(model_f(data))
            _, pred = torch.max(output, 1)
            correct_total += torch.sum(pred == label)
            data_total += data.size(0)

        pred_acc_t = correct_total.double() / data_total
    return pred_acc_t * 100




def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target_all(args):
    print(f"LLN type is {args.lln_type}, DataAug is {args.is_data_aug}")

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set base network
    # model = All_Model(args)
    domain_2_name = {
            'scannet': 'c', 'modelnet': 'm', 'shapenet': 'p'
        }
    model_f = Model.FE()
    model_f = model_f.to(device=args.device)
    model_c = Model.Pointnet_c()
    model_c = model_c.to(device=args.device)
    model_path = osp.join(args.model_root, f"seed{str(args.seed)}", args.dset, domain_2_name[args.source].upper())
    modelpath = model_path + '/source_F.pt'
    model_f.load_state_dict(torch.load(modelpath))
    modelpath = model_path + '/source_C.pt'
    model_c.load_state_dict(torch.load(modelpath))

    # ============== output file init ==============
    seed_dir = "seed" + str(args.seed)
    args.name = domain_2_name[args.source].upper() + "2" + domain_2_name[args.target].upper()
    out_file_dir = osp.join(args.output_dir, args.expname, args.dset, args.name)
    if not osp.exists(out_file_dir):
        os.makedirs(out_file_dir)
    out_file_name = (
        out_file_dir + "/" + "log_" + seed_dir + "_" + args.key_info + ".txt"
    )
    args.out_file = open(out_file_name, "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()
    # ==========================================

    # ========== log ===========
    log_dir, log_file = set_logger(args)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger, fh, sh = TxtLogger(filename=osp.abspath(osp.join(log_dir, log_file)))
    logger.info(args)
    # ==========================


    # data load
    dset_loaders = load_pointda_data(args)

    # Optimizer
    remain_epoch = 50
    params = [{
        'params': v
    } for k, v in model_f.named_parameters() if 'pred_offset' not in k]

    optimizer_g = optim.Adam(params, lr=args.lr*args.lr_F_coef, weight_decay=args.weight_decay)
    lr_schedule_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.max_epoch + remain_epoch)

    optimizer_c = optim.Adam([{'params': model_c.parameters()}],lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=args.max_epoch + remain_epoch)

    # training params
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    best = 0
    best_log_str = " "
    test_num = 0
    re_dict_to_save = {}



    # ====================================================

    # LLN method setup
    if args.is_lln:
        criterion = criterion_load(args)
        
    # CA of NVC-LLN - building feature bank and score bank (AaD)
    if args.is_ca:
        loader = dset_loaders["target"]
        num_sample = len(loader.dataset)
        fea_bank = torch.randn(num_sample, 1024)
        score_bank = torch.randn(num_sample, args.class_num).to(args.device)
        # label_bank = torch.randn(num_sample)
        pseudo_bank = torch.randn(num_sample).long()

        model_f.eval()
        model_c.eval()
        with torch.no_grad():
            iter_test = iter(loader)
            for i in range(len(loader)):
                data = next(iter_test)
                inputs = data[0][2] if args.is_data_aug or args.lln_type == "gjs" else data[0]
                indx = data[-1]
                # labels = data[1]
                inputs = inputs.to(args.device)
                output = model_f(inputs)
                outputs = model_c(output)
                output_norm = F.normalize(output)
                outputs = nn.Softmax(-1)(outputs)
                pseudo_label = torch.argmax(outputs, 1)


                fea_bank[indx] = output_norm.detach().clone().cpu()
                score_bank[indx] = outputs.detach().clone()  # .cpu()
                # label_bank[indx] = labels
                pseudo_bank[indx] = pseudo_label.detach().clone().cpu()
        model_f.train()
        model_c.train()
    


   
        
    # =====================================================
    # ================= PRE TEST ====================
    # =====================================================

    model_f.eval()
    model_c.eval()
    acc = cal_pointda(dset_loaders["test"], model_f, model_c, args.device)
    log_str = f"Task: {args.name}, Iter:{iter_num}/{max_iter}; Accuracy = {acc:.2f}% \n"
    logger.info(f"[Pre Test Info]\n{log_str}")
    model_f.train()
    model_c.train()
    # =====================================================
    # =====================================================
    loader = dset_loaders["target"]
    logger.info("================== Adaptation Begin ==================")

    
    while iter_num < max_iter:
        
        batch_loss_str = f"[Loss Info] Iter [{iter_num}/{max_iter}]: \n"

        try:
            inputs_test_all, tar_real, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test_all, tar_real, tar_idx = next(iter_test)

        if args.is_data_aug or args.lln_type == "gjs":
            if inputs_test_all[0].size(0) == 1:
                continue
        else:
            if inputs_test_all.size(0) == 1:
                continue

        if args.is_data_aug or args.lln_type == "gjs":
            inputs1, inputs2, inputs_test = (
                inputs_test_all[0].to(args.device),
                inputs_test_all[1].to(args.device),
                inputs_test_all[2].to(args.device),
            )
        else:
            inputs_test = inputs_test_all.to(args.device)
        
        # print(tar_real)

        # ========================================================
        # ============== Pseudo Label and Mask Gen ===============
        # ========================================================
        if args.is_shot and (iter_num % interval_iter == 0):
            model_f.eval()
            model_c.eval()

            args.out_file.write(
                f"\n============= Iter: {iter_num}, Update Pseudo Label ==============\n"
            )
            args.out_file.flush()
            mem_label = obtain_label_pointda(
                dset_loaders["pl"], model_f, model_c, args
            )
            mem_label = torch.from_numpy(mem_label).to(args.device)
            args.out_file.write(
                f"==================================================================\n"
            )
            args.out_file.flush()

            model_f.train()
            model_c.train()


        iter_num += 1        
           

        if args.alpha_decay:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.alpha_beta) * args.alpha
        else:
            alpha = args.alpha

        
        # ========================================================
        # =================== Model Forward ======================
        # ========================================================
        # main model forward
        
        features_test = model_f(inputs_test)
        # print(features_test)
        outputs_test = model_c(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)


        # data aug model forward
        if args.is_data_aug or args.lln_type == "gjs":
            # contras
            features_test1 = model_f(inputs1)
            outputs_test1 = model_c(features_test1)
            features_test2 = model_f(inputs2)
            outputs_test2 = model_c(features_test2)
            features = []
            features.append(features_test1)
            features.append(features_test2)
            outputs = []
            outputs.append(outputs_test1)
            outputs.append(outputs_test2)
            

 

        # ========================================================
        # ======================== loss ==========================
        # ========================================================
        loss = torch.tensor(0.0).to(args.device)

        # generate pseudo-pred
        if args.is_shot:
            batch_loss_str +=  f"PL: use shot generated... \n"
            pred = mem_label[tar_idx].long()
        else:
            batch_loss_str +=  f"PL: use self-training generated... \n"
            with torch.no_grad():
                pred = softmax_out.max(1)[1].long()  # .detach().clone()

        # sample denoising
        if args.lln_mask:
            entropy = Entropy(softmax_out.detach().clone())
            bs_weight = 1 - entropy / np.log(args.class_num)
            bs_weight = bs_weight / bs_weight.max()
            batch_loss_str += f"Batch Weight included, {bs_weight};\n"
        else:
            bs_weight = torch.ones(softmax_out.size(0)).to(args.device)

        

        # shot baseline
        if args.is_shot:
            shot_ssl_loss = nn.CrossEntropyLoss()(outputs_test, pred.long())
            shot_ssl_loss = shot_ssl_loss * args.shot_ssl_coef
            loss += shot_ssl_loss
            batch_loss_str += (
                f"[SHOT-PL] (*{args.shot_ssl_coef}): {shot_ssl_loss.item()}; \n"
            )
            # entropy
            entropy_loss = torch.mean(
                Entropy(softmax_out) * bs_weight 
            )
            entropy_loss = entropy_loss * args.shot_mi_coef
            loss += entropy_loss
            batch_loss_str += (
                f"[SHOT-MI-Entropy] (*{args.shot_mi_coef}): {entropy_loss.item()}; \n"
            )

            # diversity
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = -(
                2 * torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            )
            gentropy_loss = gentropy_loss * args.shot_mi_coef
            loss += gentropy_loss
            batch_loss_str += (
                f"[SHOT-MI-div] (*{args.shot_mi_coef}): {gentropy_loss.item()}; \n"
            )
           

        # CA NVC-LLN
        if args.is_ca:
            # update neighbor info
            with torch.no_grad():
                output_f_norm = F.normalize(features_test)
                output_f_ = output_f_norm.cpu().detach().clone()
                if args.smooth_ca > 0:
                    batch_loss_str += f"[Contrastive - Pos]: Smooth Coef (zeta_mem) {args.smooth_ca} for last epo |"
                    smooth_softmax_out_ = args.smooth_ca * score_bank[
                        tar_idx
                    ] + (1 - args.smooth_ca) * torch.clamp(
                        softmax_out.detach().clone(), 1e-4, 1.0 - 1e-4
                    )
                    smooth_softmax_out_ = (smooth_softmax_out_) / (
                        smooth_softmax_out_
                    ).sum(dim=1, keepdim=True)
                    score_bank[tar_idx] = smooth_softmax_out_
                else:
                    batch_loss_str += f"[Contrastive - Pos]: No Smooth |"
                    score_bank[tar_idx] = (
                        nn.Softmax(dim=-1)(outputs_test).detach().clone()
                    )
                fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            

                distance = output_f_ @ fea_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]  # batch x K x C

            # POS sample
            # nn
            softmax_out_un = softmax_out.unsqueeze(1).expand(
                -1, args.K, -1
            )  # batch x K x C

            pos_loss = torch.mean(
                (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
            )  
            loss += pos_loss

            batch_loss_str += f" loss is: {pos_loss.item()}; \n"
     

            
            mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0])).to(
                args.device
            )
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag

            copy = softmax_out.T  # .detach().clone()#
            dot_neg = softmax_out @ copy  # batch x batch
            dot_neg = (dot_neg * mask).sum(-1)  # batch
           
            # dot_neg = (dot_neg * mask).sum(-1)  # batch
            neg_pred = torch.mean(dot_neg)
            neg_loss = neg_pred * alpha
            loss += neg_loss
            batch_loss_str += f"[Contrastive - Neg] (*{alpha}): {neg_pred.item()} * {alpha} = {neg_loss.item()}; \n"
        

        # lln
        if args.is_lln and iter_num >= int(max_iter * args.lln_start):
            lln_output = outputs if args.lln_type == "gjs" else outputs_test
            if args.lln_mask:
                lln_loss = criterion(tar_idx, lln_output, pred.long(), bs_weight)
                batch_loss_str += (
                f"[Corrected-LLN-{args.lln_type}]: {lln_loss.item()} * {args.lln_coef}; \n"
                )
            else:
                lln_loss= criterion(tar_idx, lln_output, pred.long())
                batch_loss_str += (
                f"[LLN-{args.lln_type}]: {lln_loss.item()} * {args.lln_coef}; \n"
                )
            loss += lln_loss * args.lln_coef
            # log
            

        if args.is_data_aug:
            data_aug_tar = nn.Softmax(dim=-1)(outputs_test.detach().clone() / args.data_aug_temp)
            dataaug_loss1 = cross_entropy(data_aug_tar, outputs[0], args)
            dataaug_loss2 = cross_entropy(data_aug_tar, outputs[1], args)
            dataaug_loss = args.data_aug_coef * (
                0.5 * dataaug_loss1 + 0.5 * dataaug_loss2
            )
            batch_loss_str += (
            f"[DataAug - CE]: (*{args.data_aug_coef}=){dataaug_loss.item()}; \n"
            )
            loss += dataaug_loss


       
        optimizer_g.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer_g.step()
        optimizer_c.step()
        
        cur_epoch = iter_num // len(dset_loaders["target"])
        remainder = iter_num % len(dset_loaders["target"])
        if remainder == 1:
            lr_schedule_g.step(epoch=cur_epoch)
            lr_schedule_c.step(epoch=cur_epoch)
            print(f"Current model_f learning rate: {optimizer_g.param_groups[0]['lr']}")
            print(f"Current model_c learning rate: {optimizer_c.param_groups[0]['lr']}")
        

        # ========================================================
        # ======================== Evaluation ====================
        # ========================================================
        test_flag = (iter_num < args.warmup_eval_iter_num) and (iter_num >= (args.warmup_eval_iter_num - 100))
        if (iter_num % interval_iter == 0) or (iter_num == max_iter) or test_flag:
            # log
            logger.info(batch_loss_str)
            model_f.eval()
            model_c.eval()
            acc = cal_pointda(dset_loaders["test"], model_f, model_c, args.device)
            log_str = f"Task: {args.name}, Iter:{iter_num}/{max_iter}; Accuracy = {acc:.2f}% \n"

            args.out_file.write(
                f"\n============= Iter: {iter_num}, Evaluate Current Result ==============\n{log_str}\n==================================================================\n"
            )
            args.out_file.flush()
            if acc > best:
                best = acc
                best_log_str = log_str
            model_f.train()
            model_c.train()

            logger.info(f"[Test Info]\n{log_str}")
            test_num += 1



    best_acc_str = f"Adaptation Training End, src/tar:[{args.source}/{args.target}]\nbest acc is {str(best)}"
    print(f"best acc is {best}\n")
    print("*" * 30)

    logger.info(f"\n{best_acc_str}\nBest Log Info is: {best_log_str}\n")
    args.out_file.write(f"\n{best_acc_str}\nBest Log Info is: {best_log_str}\n")
    args.out_file.flush()

    if args.is_var_noise_eval:
        # dict saving
        dict_name = log_file.split('.')[0] + '.pkl'
        dict_f_name = osp.abspath(osp.join(out_file_dir, dict_name))
        with open(dict_f_name, 'wb') as tf:
            pickle.dump(re_dict_to_save, tf)


    # remove current file handler to avoid log file error
    logger.removeHandler(fh)
    # remove current stream handler to avoid IO info error
    logger.removeHandler(sh)

    return '.', best.item(), best_log_str


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="SHOT")
    parser.add_argument(
        "--config",
        type=str,
        help="config file, None for not using, eg,./config/xxx.yml",
        default=None,
    )
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="PointDA_data",
        choices=["VISDA-C", "office", "office-home", "domainnet", "PointDA_data"],
    )

    parser.add_argument(
        "--list_name",
        type=str,
        default="image_list",
    )
    parser.add_argument("--net", type=str, default="resnet50", help="resnet50, res101")
    parser.add_argument("--net_mode", type=str, choices=["fbc", "fc"], default="fbc")
    parser.add_argument("-s", "--source", default="c", help="source domain(s)")
    parser.add_argument("-t", "--target", default="s", help="target domain(s)")

    parser.add_argument("--max_epoch", type=int, default=100, help="max iterations")
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=6, help="number of workers")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--da", type=str, default="uda", choices=["uda", "pda"])

    # DIR related
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="root path of DATASOURCE, eg, ./DATASOURCE",
        default="./DATASOURCE",
    )
    parser.add_argument(
        "--model_root",
        help="model load and save root dir",
        default="./Models/pointda",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output"
    )
    parser.add_argument(
        "--s_model", help="source model suffix name, e.g., new", default=None
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs"
    )
    parser.add_argument("--expname", type=str, default="tpami")
    parser.add_argument("--key_info", type=str, default="pointda_test")

    # lr
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument(
        "--lr_F_coef", type=float, default=0.5, help="learning rate for backbone"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="optimiser weight decay"
    )


    
    ###################################################################################
    
    # SHOT
    parser.add_argument(
        "--is_shot",
        type=ast.literal_eval,
        default=False,
        help="whether use shot method to generate pseudo-label",
    )
    parser.add_argument(
        "--shot_ssl_coef", default=0.3, type=float, help="shot ssl coef"
    )
    parser.add_argument(
        "--shot_mi_coef", default=1.0, type=float, help="mutual info loss coef"
    )
    parser.add_argument("--epsilon", type=float, default=1e-5)
    # pseudo-label
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="threshold for PL generating, to avoid unpredicted cls",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="cosine",
        choices=["euclidean", "cosine"],
        help="distance metric for calculate the point to intric",
    )

    # lln losses
    parser.add_argument("--is_lln", type=ast.literal_eval, default=True)
    parser.add_argument("--beta", default=0.6, type=float, help="zeta_lln for time step t")
    parser.add_argument(
        "--lamb", default=0.1, type=float, help="tuning param for elr coef, equivalent to lln_coef while lln_type == elr"
    )
    parser.add_argument(
        "--lln_type",
        type=str,
        choices=["sl", "gjs", "elr", "gce", "baseline"],
        default="elr",
    )
    parser.add_argument("--lln_coef", default=0.0001, type=float, help="lln coef")
    parser.add_argument(
        "--lln_start", default=0, type=float, help="lln loss start portion"
    )
    
    # NVC-LLN
    # AaD - orgCA
    parser.add_argument("--is_ca", type=ast.literal_eval, default=True)
    parser.add_argument("--K", type=int, default=2)  
    parser.add_argument("--alpha", type=float, default=0.1) 
    parser.add_argument("--alpha_beta", type=float, default=5.0)  
    parser.add_argument("--alpha_decay", type=ast.literal_eval, default=True)
    # smooth CA - NVC-LLN
    parser.add_argument("--smooth_ca", type=float, default=0.5, help="zeta_mem for time step t used for neighbor prediction logits. -1 means no smooth.")
    # sample denoising - NVC-LLN
    parser.add_argument("--lln_mask", type=ast.literal_eval, default=True, help="whether use sample denoising for lln loss correction")
    # data aug - NVC-LLN
    parser.add_argument(
        "--is_data_aug",
        type=ast.literal_eval,
        default=True,
        help="whether use data augmentation for NVC-LLN and GJS",
    )
    parser.add_argument(
        "--data_aug_coef", default=2.0, type=float, help="lamb_aug, data augmentation coef in NVC-LLN"
    )
    parser.add_argument(
        "--data_aug_temp", default=1.0, type=float, help="data aug temperature, tau"
    )
    
    # evaluation
    # variance reduction effect and denoise effect
    parser.add_argument(
        "--is_var_noise_eval",
        type=ast.literal_eval,
        default=False,
        help="whether to evaluate the variance reduction and denoising effect",
    )
    parser.add_argument(
        "--warmup_eval_iter_num",
        default=20,
        type=int,
    )
    
    
    args = parser.parse_args()

    if args.config is not None:
        cfg_dir = osp.abspath(osp.join(osp.dirname(__file__), "../../", args.config))
        opt = vars(args)
        args = yaml.load(open(cfg_dir), Loader=yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)

    args.log_dir = osp.abspath(osp.expanduser(args.log_dir))
    args.root = osp.abspath(osp.expanduser(args.root))
    args.model_root = osp.abspath(osp.expanduser(args.model_root))
    args.output_dir = osp.abspath(osp.expanduser(args.output_dir))

    
    # if args.dset == "PointDA_data":
    task = ['scannet', 'modelnet', 'shapenet']
    names = ['c', 'm', 'p']
    domain_2_name = {
        'scannet': 'c', 'modelnet': 'm', 'shapenet': 'p'
    }
    args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.key_info = (
        args.key_info + "_" + args.s_model
        if args.s_model is not None
        else args.key_info
    )
    
    
    train_target_all(args)
