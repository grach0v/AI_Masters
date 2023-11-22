from torch.optim.lr_scheduler import MultiStepLR
from torchviz                 import make_dot

import numpy as np

import argparse
import torch
import time
import os

from voc_dataloader      import get_train_dataloader, get_test_dataloader

# from model_resnet18  import SSD_ResNet18
from model_vgg16     import SSD_VGG16
from multibox_loss   import MultiBoxLoss
from prior_boxes     import prior_boxes
from logger          import Logger

from tqdm import tqdm

from map import mean_average_precision

def train_process(custom_config, args, model_path=None):
    torch.manual_seed(     args['seed'])
    np.random.seed   (seed=args['seed'])

    dataset_root_dir          = args['dataset_root_dir']
    train_annotation_filename = os.path.join( dataset_root_dir, "ImageSets/Main/trainval.txt" )
    test_annotation_filename  = os.path.join( dataset_root_dir, "ImageSets/Main/test.txt"     )
    train_dataloader          = get_train_dataloader(dataset_root_dir, train_annotation_filename, args['batch_size'], args['num_workers'])
    test_dataloader           = get_test_dataloader (dataset_root_dir, test_annotation_filename , args['batch_size'], args['num_workers'])

    learning_rate = args['learning_rate']

    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    model = SSD_VGG16(custom_config['num_priors'], custom_config['num_classes'])
    
    if model_path != None:
        with open(model_path, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(state_dict)

    prior_box_s = prior_boxes(custom_config)
    prior_box_s_gpu = prior_box_s .cuda()

    overlap_threshold = custom_config['overlap_threshold']
    neg_pos_ratio     = custom_config['neg_pos_ratio'    ]
    variance          = custom_config['variance']

    criterion = MultiBoxLoss( overlap_threshold, neg_pos_ratio, variance)
    model    .cuda()
    criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args['multistep'], gamma=0.2)

    best_loc_loss, best_cls_loss, best_loss = np.inf, np.inf, np.inf

    mean_average_precision_hist = [] 

    for epoch in tqdm(list(range(args['epochs']))):
        #Train model
        train_loc_loss, train_cls_loss, train_loss = 0, 0, 0
        model.train()
        for i, (image_s_cpu, box_ss_cpu, label_ss_cpu) in tqdm(enumerate(train_dataloader)):

            image_s_gpu  = image_s_cpu.cuda() #внизу боксы тоже перебросить на гпу
            label_ss_gpu = [ label_s_cpu.cuda() for label_s_cpu in label_ss_cpu ]
            box_ss_gpu   = [ box_s_cpu.cuda() for box_s_cpu   in box_ss_cpu   ]

            pred_loc_ss_gpu, pred_conf_ss_gpu = model(image_s_gpu)

            loc_loss, cls_loss = criterion( (pred_loc_ss_gpu, pred_conf_ss_gpu, prior_box_s_gpu), (label_ss_gpu, box_ss_gpu))
            loss = loc_loss + cls_loss

            optimizer.zero_grad()
            loss     .backward()
            optimizer.step()

            train_loc_loss += loc_loss.item()
            train_cls_loss += cls_loss.item()
            train_loss     += loss    .item()
            torch.cuda.empty_cache()

        scheduler.step()
        #Eval model
        eval_loc_loss, eval_cls_loss, eval_loss = 0, 0, 0
        model.eval()
        torch.cuda.empty_cache()

        maps = []
        for i , (image_s_cpu, box_ss_cpu, label_ss_cpu) in enumerate(test_dataloader):
            image_s_gpu = image_s_cpu.cuda()
            label_ss_gpu = [ label_s_cpu.cuda() for label_s_cpu in label_ss_cpu ]
            box_ss_gpu   = [ box_s_cpu.cuda() for box_s_cpu   in box_ss_cpu   ]

            pred_loc_ss_gpu, pred_conf_ss_gpu = model(image_s_gpu)

            map = mean_average_precision(
                pred_loc_ss_gpu.detach().cpu(),
                pred_conf_ss_gpu.detach().cpu(),
                box_ss_cpu,
                label_ss_cpu,
                custom_config['overlap_threshold']
            )

            maps.append(map)
            
            loc_loss, cls_loss = criterion( 
                (pred_loc_ss_gpu, pred_conf_ss_gpu, prior_box_s_gpu), 
                (label_ss_gpu, box_ss_gpu)
            )

            loss = loc_loss + cls_loss

            eval_loc_loss += loc_loss.item()
            eval_cls_loss += cls_loss.item()
            eval_loss     += loss    .item()
            torch.cuda.empty_cache()

        maps = np.mean(maps)
        mean_average_precision_hist.append(maps)
        
        print(
            'epoch[{}] | lr {:.5f} | loc_loss [{:.2f}/{:.2f}] | cls_loss [{:.2f}/{:.2f}] | total_loss [{:.2f}/{:.2f}] | map {:.4f}'
            .format(epoch, scheduler.get_last_lr()[0], train_loc_loss, eval_loc_loss, train_cls_loss, eval_cls_loss, train_loss, eval_loss, maps)
        )

        if eval_loss < best_loss :
            torch.save(model.state_dict(), os.path.join(args['output'], f"{custom_config['model_name']}{eval_loss:.2f}.pth"))
            best_loc_loss, best_cls_loss, best_loss = eval_loc_loss, eval_cls_loss, eval_loss
    
    return model, model.state_dict()

