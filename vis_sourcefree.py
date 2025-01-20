import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.utils.weight_norm as weightNorm
import os.path as osp

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name='resnet101', pretrained=False):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=pretrained)


        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="bn"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        out=x
        return out



class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="wn"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class ResNet_FE(nn.Module):
    def __init__(self):
        super().__init__()
        model_resnet = torchvision.models.resnet50(True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.avgpool)
        self.bottle = nn.Linear(2048, 256)
        self.bn = nn.BatchNorm1d(256)

    def forward(self, x):
        out = self.feature_layers(x)
        out = out.view(out.size(0), -1)
        out = self.bn(self.bottle(out))
        return out
    

class All_Model(nn.Module):
    def __init__(self, args, load_ensemble_seed=None):
        super(All_Model, self).__init__()

        self.net_mode = args.net_mode
        
        # network build
        if args.net_mode == 'fbc':
            self.netF = ResBase(res_name=args.net).to(args.device)
            self.netB = feat_bootleneck(feature_dim=self.netF.in_features).to(args.device)
            self.netC = feat_classifier(class_num=args.class_num).to(args.device)
        elif args.net_mode == 'fc':
            self.netF = ResNet_FE().to(args.device)
            self.netC = feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=256).to(args.device)
        else:
            print(f"Unknown net mode: {self.net_mode}")
        
        # weight load
        if load_ensemble_seed is None:
            self.weight_load(args)
        else:
            self.weight_load_ensemble(args, load_ensemble_seed)
            

    def forward(self, img):
        if self.net_mode == 'fbc':
            fea = self.netB(self.netF(img))
            out = self.netC(fea)
            
        elif self.net_mode == 'fc':
            fea = self.netF(img)
            out = self.netC(fea)
        else:
            print(f"Unknown net mode: {self.net_mode}")
        return fea, out
    
    def weight_load(self, args):
        weights_dir = args.source.upper(
        )[0] + '_' + args.s_model if args.s_model is not None else args.source.upper()[0]

        seed_dir = 'seed' + str(args.seed)
        
        self.netF.load_state_dict(torch.load(
            osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_F.pt')))
        self.netC.load_state_dict(torch.load(
            osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_C.pt')))
        if self.net_mode == 'fbc':
            self.netB.load_state_dict(torch.load(
                osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_B.pt')))
        
        return None
    
    def weight_load_ensemble(self, args, seed):
        weights_dir = args.source.upper(
        )[0] + '_' + args.s_model if args.s_model is not None else args.source.upper()[0]

        seed_dir = 'seed' + str(seed)
        
        self.netF.load_state_dict(torch.load(
            osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_F.pt')))
        self.netC.load_state_dict(torch.load(
            osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_C.pt')))
        if self.net_mode == 'fbc':
            self.netB.load_state_dict(torch.load(
                osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_B.pt')))
        
        return None
            

    
    def weight_reload_partial(self, args, reload_mode=-1):
        
        weights_dir = args.source.upper(
        )[0] + '_' + args.s_model if args.s_model is not None else args.source.upper()[0]
        seed_dir = 'seed' + str(args.seed)
        
        full_netF_param = torch.load(osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_F.pt'))
        
        if reload_mode in [1, 2]:
            # reload bottle - bn - cls
            print(f"Reload mode {reload_mode}, loading bottleneck and classifier...")
            self.netC.load_state_dict(torch.load(
            osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_C.pt')))
            for param in self.netC.parameters():
                param.requires_grad = True
            if self.net_mode == 'fbc':
                self.netB.load_state_dict(torch.load(
                    osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_B.pt')))
                for param in self.netB.parameters():
                    param.requires_grad = True
            else:
                net_F_b_dict = {k: v for k, v in full_netF_param.items() if ('netF.bottle.' in k) or ('netF.bn.' in k)}
                self.netF.load_state_dict(net_F_b_dict, strict=False)
                for name, param in self.netF.named_parameters():
                    if name in net_F_b_dict:
                        param.requires_grad = True
            
            if reload_mode == 1:
                print(f"Reload mode {reload_mode}, loading layer 3 and layer 4...")
                # layer 3 - layer 4 
                net_F_dict = {k:v for k, v in full_netF_param.items() if ('layer3' in k) or ('layer4' in k)}
                self.netF.load_state_dict(net_F_dict, strict=False)
                for name, param in self.netF.named_parameters():
                    if name in net_F_dict:
                        param.requires_grad = True
                
        else:
            # no reload
            print(f'Unknown reload mode {reload_mode}, pass...')
    
class All_Model_PES(nn.Module):
    def __init__(self, args, load_ensemble_seed=None):
        super(All_Model, self).__init__()

        self.net_mode = args.net_mode
        
        # network build
        if args.net_mode == 'fbc':
            self.netF = ResBase(res_name=args.net).to(args.device)
            self.netB = feat_bootleneck(feature_dim=self.netF.in_features).to(args.device)
            self.netC = feat_classifier(class_num=args.class_num).to(args.device)
        elif args.net_mode == 'fc':
            self.netF = ResNet_FE().to(args.device)
            self.netC = feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=256).to(args.device)
        else:
            print(f"Unknown net mode: {self.net_mode}")
        
        # weight load
        if load_ensemble_seed is None:
            self.weight_load(args)
        else:
            self.weight_load_ensemble(args, load_ensemble_seed)
            

    def forward(self, img):
        if self.net_mode == 'fbc':
            fea = self.netB(self.netF(img))
            out = self.netC(fea)
            
        elif self.net_mode == 'fc':
            fea = self.netF(img)
            out = self.netC(fea)
        else:
            print(f"Unknown net mode: {self.net_mode}")
        return fea, out
    
    def weight_load(self, args):
        weights_dir = args.source.upper(
        )[0] + '_' + args.s_model if args.s_model is not None else args.source.upper()[0]

        seed_dir = 'seed' + str(args.seed)
        
        self.netF.load_state_dict(torch.load(
            osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_F.pt')))
        self.netC.load_state_dict(torch.load(
            osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_C.pt')))
        if self.net_mode == 'fbc':
            self.netB.load_state_dict(torch.load(
                osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_B.pt')))
        
        return None
    
    def weight_reload_partial(self, args, reload_mode=-1):
        
        weights_dir = args.source.upper(
        )[0] + '_' + args.s_model if args.s_model is not None else args.source.upper()[0]

        seed_dir = 'seed' + str(args.seed)
        
        full_netF_param = torch.load(osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_F.pt'))
        
        if reload_mode in [1, 2]:
            # reload bottle - bn - cls
            print(f"Reload mode {reload_mode}, loading bottleneck and classifier...")
            self.netC.load_state_dict(torch.load(
            osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_C.pt')))
            if self.net_mode == 'fbc':
                self.netB.load_state_dict(torch.load(
                    osp.abspath(f'{args.model_root}/{seed_dir}/{args.dset}/{weights_dir}/source_B.pt')))
                
            else:
                net_F_b_dict = {k: v for k, v in full_netF_param.items() if ('netF.bottle.' in k) or ('netF.bn.' in k)}
                self.netF.load_state_dict(net_F_b_dict, strict=False)
            
            if reload_mode == 1:
                
                print(f"Reload mode {reload_mode}, loading layer 3 and layer 4...")
                # layer 3 - layer 4 
                net_F_dict = {k:v for k, v in full_netF_param.items() if ('layer3' in k) or ('layer4' in k)}
                self.netF.load_state_dict(net_F_dict, strict=False)
                
            
        else:
            # no reload
            print(f'Unknown reload mode {reload_mode}, pass...')
        
            
            

            
    