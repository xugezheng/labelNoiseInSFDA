# labelNoiseInSFDA
Official Implementation for our Paper "Unraveling the Mysteries of Label Noise in Source-Free Domain Adaptation: Theory and Practice".

## Repo Structure

```
labelNoiseInSFDA/
├── common/                                 # Datasets 
│   ├── vision/
│   │   └── datasets/
├── config/                                 # yml examples
├── DATASOURCE/                             # data
│   ├── domainnet/
|   |   ├── image_list/
│   │   └── ...
│   ├── office/
│   ├── office-home/
│   └── VISDA-C/
├── Models/                                 # source model
├── pointDA-10/                             # adaptation code for PointDA-10 dataset
│   └── train_tar_merged.py
├── README.md
├── train_source_imbalance.py
├── train_target_labelNoise_SFDA.py         # main adaptation script
├── loggers.py
├── utils_dataloader.py
├── utils_evaluation.py
├── utils_lln_losses.py
├── utils_loss.py
├── utils_pseudo_label.py
├── utils_source_train.py
└── vis_sourcefree.py                       # network
```

## Get Started

### 1. Data Preparation

For office-31, please download the data from [here](https://drive.google.com/file/d/1dmEhhzoP-dnVOqsV_zDUNRKUy4dKN6Bl/view?usp=sharing). Unzip the `office.zip` into `./DATASOURCE/`. Make sure the `image_list` folder is in the `./DATASOURCE/office/`.

For office-home dataset, please download the data from [here](https://drive.google.com/file/d/1gRSeLmBTKWjSiqe6jRWaNsXTqzDdqKPr/view?usp=sharing). Unzip the `office-home.zip` into `./DATASOURCE/`. Make sure the `image_list_nrc` (for office-home) and `image_list_partial` (for partial set) are in the `./DATASOURCE/VISDA-C/` folder.

For VisDA2017 dataset, the original data can be downloaded from [here](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification). Extract `train.tar` and `validation.tar` to ./DATASOURCE/VISDA-C. Make sure the `image_list` (for VisDA2017) and `image_list_imb` (for VisDA-RUST provided by [I-SFDA](https://github.com/LeoXinhaoLee/Imbalanced-Source-free-Domain-Adaptation)) are in the `./DATASOURCE/VISDA-C/` folder.

For DomainNet dataset, the original dataset can be downloaded from [here](https://ai.bu.edu/M3SDA/). Please extract the image data of each `DOMAIN` into `./DATASOURCE/domainnet/DOMAIN`. Make sure the `image_list` folder is in the `./DATASOURCE/domainnet/` (for 40 categories classification task).

For pointDA-10 dataset can be download from [here](https://github.com/canqin001/PointDAN/tree/master?tab=readme-ov-file), provided in [PointDAN](https://github.com/canqin001/PointDAN) repo. Please unzip the data file into `./DATASOURCE/`, keeping its original folder name (`PointDA_data`).

---

Make sure the provided `DATASOURCE` folder and the related `image_list*` folders are well downloaded.

### 2. Models Preparation

For Office-Home and VisDA datasets, we utilized the source models provided by previous Repos (including [SHOT](https://github.com/tim-learn/SHOT) and [NRC](https://github.com/Albert0147/NRC_SFDA)).

For Office-31 and Domainnet(40 categories classification task), we trained the source models by ourself following [SHOT](https://github.com/tim-learn/SHOT).

For PointDA, we follow the instructions from [NRC](https://github.com/Albert0147/NRC_SFDA) and [PointDAN](https://github.com/canqin001/PointDAN) to train the source models.

All model checkpoints are provided [here](https://drive.google.com/drive/folders/1SfPPaTu69ef4TAuaNeW6KSd7bMlqyg6l?usp=sharing). Please download the model checkpoints into `./Models/`.

### 3. Adaptation Scripts

#### 3.1 Image classification task

We provide two adaptation configuration methods, through command line and yml config file. Here, we take office-home dataset as an example.

1. command line
   - SHOT + ELR (or other LLN losses)
     ```shell
     python train_target_labelNoise_SFDA.py \
       --dset office-home --source Ar --target Rw --net resnet50 --net_mode fc --list_name image_list_nrc \
       --max_epoch 45 --interval 45 \
       --root ./DATASOURCE \
       --model_root ./Models/officehome \
       --output_dir ./output \
       --log_dir ./logs \
       --expname tpami --key_info officehome_test_shot_elr --seed 2021 \
       --lr_decay True --lr_decay_type shot --lr_F_coef 0.5 --weight_decay 0.0005 \
       --is_shot True \ #SHOT
       --is_lln True --lln_type elr --beta 0.6 --lamb 3.0 --lln_coef 0.3 \ # params for LLN losses, such as sl, gce, gjs
     ```
   - NVC-LLN
       ```shell
       python train_target_labelNoise_SFDA.py \
       --dset office-home --source Ar --target Rw --net resnet50 --net_mode fc --list_name image_list_nrc \
       --max_epoch 45 --interval 45 \
       --root ./DATASOURCE \
       --model_root ./Models/officehome \
       --output_dir ./output \
       --log_dir ./logs \
       --expname tpami --key_info officehome_test_nvc_gce --seed 2021 \
       --lr_decay True --lr_decay_type shot --lr_F_coef 0.5 --weight_decay 0.0005 \
       --is_lln True --lln_type gce --beta 0.2 --lln_coef 0.3 --lln_mask True \ # C-LLN losses, such as sl, gce, gjs, elr
       --is_ca True --K 3 --alpha 1 --alpha_beta 0.75 --alpha_decay False --smooth_ca 0.8 \ # smooth CA
       --is_data_aug True --data_aug_coef 0.1 --data_aug_temp 1.0 \ # data aug alignment
       ```

2. yml file

    > YML config is prior to command line args

    ```shell
    python train_target_labelNoise_SFDA.py --config ./config/oh_nvcgce_example.yml
    ```

#### 3.2 3D point cloud task

The code for pointda-10 dataset is built on [NRC](https://github.com/Albert0147/NRC_SFDA) and [PointDAN](https://github.com/canqin001/PointDAN). The parameters for ELR and NVC-LLN are same as those in `train_target_labelNoise_SFDA`. To conduct adaptation, please the script `./pointDA-10/train_tar_merged.py`

<!-- ## Acknowledgement and citation -->

