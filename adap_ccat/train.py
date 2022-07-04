import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, optim

from os.path import join as osj
# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model_utils import save_model, load_model, set_optimizer
from utils import yaml_config_hook
from dataset2 import dataset_DFD
from simclr.modules import SingleViT as VViT
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
from datetime import datetime
from sklearn import metrics
from setLogger import *
import pandas as pd

from torchsampler import ImbalancedDatasetSampler # imbalance

def decisionMaker(args, pred):
    
    if 'avg' in args.testMode:
        if isinstance(pred, list):
            pred = torch.vstack(pred)
            pred =torch.mean(pred, 0)
        else:
            pred = pred[0]  #
        pred =torch.argmax(pred).cpu().numpy()
    elif 'top' in args.testMode:
        k = int(args.testMode.split('-')[-1])
        pred = torch.vstack(pred)
        ratio = pred[:,1]-pred[:,0]
        ratio_amp = torch.sort(torch.abs(ratio), descending=True)
        ind = ratio_amp.indices
        ratio = torch.mean(ratio[ind[:k]])
        pred = 1 if ratio>=0 else 0
    
    return pred




def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset
    train_dataset = dataset_DFD(0, args, mode = 'train')
    test_dataset = dataset_DFD(0, args, mode='test')


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=ImbalancedDatasetSampler(train_dataset, labels=train_dataset.label),
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers if args.test_aug!=1 else 0,
    )
    #batch_size=args.batch_size if args.test_aug!=1 else 1,


    # Model
    model = VViT(device=args.device, args=args, mode='train')
    model = model.to(args.device)
    # model.load_state_dict(torch.load(args.model_path))
    model.train()

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = set_optimizer(args, model)

    # Init logging inforamtion
    output_path = osj(args.result_path, args.exp_name)
    global_step = 1
    total_steps = args.total_steps
    metric_dic = {}
    best_metric_dic = {}
    for key in ['global_step', 'accuracy', 'recall', 'precision', 'f1', 'auc']:
        metric_dic[key] = [0]
        best_metric_dic[key] = 0
    """
    metric_dic['global_step'] = [0]
    metric_dic['accuracy'] = [0]
    """

    # Resume if exists
    global_step, model, optimizer, scheduler, metric_dic = load_model(args, model, optimizer, scheduler, global_step, metric_dic, mertric_type=None)

    # Start training
    train_iter = iter(train_loader)
    step_iterator = tqdm(range(global_step, total_steps),
                          desc="Training (X/X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    while True:
        for global_step in step_iterator:
            
            try:
                x, y = train_iter.next()
            except:
                train_iter = iter(train_loader)
                x, y = train_iter.next()
            optimizer.zero_grad()
            # (16, 3, 224, 224)
            pred=model(x, forward_mode='train')
            loss = criterion(pred.to(args.device),y.to(args.device))
            # print()
            # print(pred, y)
            # print()
            loss.backward()
            step_iterator.set_description(f"Training ({global_step}/{total_steps} Steps) (loss={loss.item():.5f})")

            optimizer.step()
            scheduler.step()

            if global_step % args.save_every == 0:
                save_model(args, model, optimizer, scheduler, global_step, metric_dic)

            if global_step % args.eval_every == 0:
                metric_dic = val(model, test_loader, global_step, total_steps, metric_dic, output_path)
                model.train()
                for key in metric_dic.keys():
                    if metric_dic[key][-1] > best_metric_dic[key]:
                        best_metric_dic[key] = metric_dic[key][-1]
                        save_model(args, model, optimizer, scheduler, global_step, metric_dic=metric_dic, mertric_type=key)


            if global_step == total_steps :
                break
        if global_step == total_steps :
            break



def val(model, test_loader, global_step, total_steps, metric_dic, output_path):
    model.eval()

    fp1, fp2 = open(osj(output_path, f'covid.csv'), 'w'), open(
        osj(output_path, f'non-covid.csv'), 'w')

    with torch.no_grad():
        preds, labs = [], []
        accs, f1, re, pr = [], [], [], []

        outfile = []

        epoch_iterator = tqdm(test_loader,
                              desc="Validation (X/X Steps)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

        for step, data in enumerate(epoch_iterator):
            if args.test_aug == 1:
                x, y = data
                # x, y = x.to(args.device), y.to(args.device)

                if isinstance(x, list):
                    pred = []

                    for x_it in x:
                        pred_inner = model(x_it, forward_mode='test')
                        pred.extend(pred_inner)
                        filename = x_it['fn']

                    pred = decisionMaker(args, pred)
                    preds.append(pred)
                else:
                    filename = x['fn']
                    pred = model(x, forward_mode='test')
                    pred = decisionMaker(args, pred)
                    preds.append(pred)

                labs.extend(y.cpu().numpy())

            else:
               # x, y, issafe = data
                x, y = data

                #print(x)
               # x, y = x.to(args.device), y.to(args.device)
            
                pred = model(x, forward_mode='test')
                filename = x['fn']
                labs.extend(y.cpu().numpy())

                preds.append(torch.argmax(pred, 1).cpu().numpy()[0])

#                 if issafe:
#                     pred = model(x, forward_mode='test')
#                     filename = x['fn']
#                     labs.extend(y.cpu().numpy())

#                     preds.append(torch.argmax(pred, 1).cpu().numpy()[0])
#                 else:
#                     filename = x['fn']
#                     pred = model(x, forward_mode='test')
#                     labs.extend(y.cpu().numpy())

#                     preds.append(torch.argmax(pred, 1).cpu().numpy()[0])
#                     LOG.info(f'Found a insufficient CT scan in {x["fn"][0]}')

            if preds[-1] == 1:
                fp1.write('%s,' % os.path.basename(filename[0]))
            else:
                fp2.write('%s,' % os.path.basename(filename[0]))

            outfile.append([os.path.basename(filename[0]), preds[-1]])
            epoch_iterator.set_description(f"Validation ({global_step}/{total_steps} Steps)")

        outfile = pd.DataFrame(outfile)
        os.makedirs(osj(output_path, "val_per_volume"), exist_ok=True)
        outfile.to_csv(osj(output_path, "val_per_volume", f'val_{global_step}.csv'), index=False)


        preds, labs = np.asarray(preds), np.asarray(labs)
        # preds = [0 ,1, 0, 1, ..]
        # labels = [0 ,1, 0, 1, ..]
        metric_dic['global_step'].append(global_step)
        metric_dic['accuracy'].append(np.mean(preds == labs))
        metric_dic['recall'].append(recall_score(labs, preds, average='macro'))
        metric_dic['precision'].append(precision_score(labs, preds, average='macro'))
        metric_dic['f1'].append(f1_score(labs, preds, average='macro'))
        fpr, tpr, thresholds = metrics.roc_curve(labs, preds, pos_label=1)
        metric_dic['auc'].append(metrics.auc(fpr, tpr)) # TODO AUC의 input은 prob이여야하는데 0, 1 이 아니

        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        LOG.info('=' * 80)
        LOG.info(f"Validation at {global_step}/{total_steps} Steps, Test time: {date_time}")
        LOG.info('=' * 80)
        metric_str = ''
        for key in metric_dic.keys():
            metric_str += f'{key}: {metric_dic[key][-1]:5f} \n'
        LOG.info(
            f'model={args.model_path}, image size={args.image_size}, crop_size={args.crop_size}, max #crops={args.max_det}, test mode={args.testMode}')
        # LOG.info(f'(Marco) Validation Accuracy={acc}, Precision={pr}, Recall={re}, F1-Score={f1}, AUC={auc}')
        LOG.info(metric_str)
        LOG.info('=' * 80)

        metric_csv = pd.DataFrame(metric_dic)
        metric_csv.to_csv(osj(output_path, 'val_metrics.csv'), index=False)

        return metric_dic


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config_train.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()


    # os.path.join
    output_path = osj(args.result_path, args.exp_name)
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    global LOG
    LOG = init_logging(log_file=osj(output_path,"test_eval.log"))
    LOG.info(args)

    main(args)
