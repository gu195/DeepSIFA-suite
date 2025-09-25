import os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
import gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.utils.data
from scipy import stats

from torch.utils.tensorboard import SummaryWriter
import time

from utils.xrayloader import XrayDataset_val
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer_v2, optimizer_kwargs
from utils.metrics_2cls import *
from collections import OrderedDict
import csv
import shutil


def get_cfg(fold):
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str)
    parser.add_argument('--n_epochs', type=int, default=1)  
    parser.add_argument('--bt_size', type=int, default=1)

    # github上路径
    parser.add_argument('--weight_path', type=str, default='./checkpoints/fold{}/best_acc.pth'.format(fold))
    parser.add_argument('--data_dir', type=str, default='D:/1DeepSIFA/data/mlkl/train/v1/归一化插值后npz_1024_高斯平滑1')
    parser.add_argument('--val_csv_dir', type=str, default='D:/1DeepSIFA/data/mlkl/train/v1/5折交叉验证_42/val_fold{}_多标签.csv'.format(fold))
    parser.add_argument('--results_dir', default='./logs_val验证/', type=str, metavar='FILENAME', help='Output csv file for validation results (summary)')
    parse_config = parser.parse_args()
    print(parse_config)
    return parse_config


def compute_metrics(outputs, targets, loss_fn):
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()

    print()
    print('compute_metrics_outputs:',outputs)
    print('compute_metrics_targets:',targets)
    targets_ = torch.argmax(targets, dim=1)
    loss = loss_fn(outputs, targets_).cpu().item()
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall_F = Recall(outputs, targets)
    precision = Precision(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    specificity = spe(outputs, targets)
    metrics = OrderedDict([
        ('loss', loss),
        ('acc', acc),
        ('f1', f1),
        ('recall', recall_F),
        ('precision', precision),
        ('kappa', kappa),
        ('confusion_matrix',cm),
        ('specificity',specificity)
    ])
        
    return metrics


# -------------------------- test func --------------------------#
def test(epoch, model, loader_eval, loss_fn, fold):
    print("-------------testing-----------")
    model.eval()
    predictions = []
    allscore = []
    labels = []
    nameFP = []
    nameFN = []
    nameTP1 = []
    nameTP2 = []
    nameTN = []
    name_low = []


    with torch.no_grad():
        # 初始化错误预测列表
        nameFP.append('第{}折'.format(fold))
        nameFP.append('name label pre')
        nameFN.append('第{}折'.format(fold))
        nameFN.append('name label pre')
        nameTP1.append('第{}折'.format(fold))
        nameTP1.append('name label pre')
        nameTP2.append('第{}折'.format(fold))
        nameTP2.append('name label pre')
        nameTN.append('第{}折'.format(fold))
        nameTN.append('name label pre')
        for img, label, label_onehot, name in tqdm(loader_eval):
            img = img.cuda().float()
            label = label.cuda()
            label_onehot = label_onehot.cuda()
            output = model(img)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output_softmax = F.softmax(output, dim=1)

            max_probs, max_indices = torch.max(output_softmax, dim=1)
            if max_indices == 1:
                realprobs = max_probs
            else:
                realprobs = 1 - max_probs
            # 用字典保存结果
            result_dict = {
                'name': name[0],
                'label': label.item(),  
                'score': round(realprobs.item(), 2),
                'pre': max_indices.item()
            }
            allscore.append(result_dict) 
            predictions.append(output)
            labels.append(label_onehot)

            if round(max_probs.item(), 2) >0.7:

                # 记录预测错误的文件名
                preds = torch.argmax(output, dim=1)
                if label == 0 and (preds == 1 or preds == 2):
                    name_message = [name[0], label.item(), preds.item()]
                    nameFP.append(name_message)

                # 记录预测错误的文件名,把异常预测为正常
                if label == 1  and preds == 0:
                    name_message = [name[0], label.item(), preds.item()]
                    nameFN.append(name_message)

                # 记录预测正确的文件名，且是预测为1
                if label == 1 and preds == 1:
                    name_message = [name[0], label.item(), preds.item()]
                    nameTP1.append(name_message)

                if label == 1 and preds == 2:
                    name_message = [name[0], label.item(), preds.item()]
                    nameTP2.append(name_message)

                if label == 0 and preds == 0:
                    name_message = [name[0], label.item(), preds.item()]
                    nameTN.append(name_message)
            else:
                preds = torch.argmax(output, dim=1)
                name_message = [name[0], label.item(), preds.item()]
                name_low.append(name_message)


    evaluation_metrics = compute_metrics(predictions, labels, loss_fn)
    # 初始化两个列表
    subscore1 = []
    subscore2 = []

    # 遍历 allscore 列表
    for result in allscore:
        if result['score'] > 0.7:
            subscore1.append(result)
            predictions.append(result['pre'])
            labels.append(result['label'])
        else:
            subscore2.append(result)

    return evaluation_metrics, nameFP, nameFN, nameTP1, nameTP2, nameTN, name_low, subscore1, subscore2, allscore


def get_ci(global_list):
    average_value = round(np.mean(global_list), 3)
    standard_error  = stats.sem(global_list)
    a = round(average_value - 1.96 * standard_error, 3)
    b = round(average_value + 1.96 * standard_error, 3)
    return [average_value, (a, b)]

if __name__ == '__main__':
    global_nameFN = []
    global_nameFP = []
    global_nameTP1 = []
    global_nameLOW = []
    global_nameTN = []
    global_accuracy = []
    global_sensitivity = [] 
    global_F1_Score = []
    for fold in range(1,6):
        # -------------------------- get args --------------------------#
        gc.collect()
        torch.cuda.empty_cache()
        parse_config = get_cfg(fold)

        # -------------------------- build dataloaders --------------------------#
        dataset = XrayDataset_val(parse_config)

        loader_eval = torch.utils.data.DataLoader(dataset, batch_size=parse_config.bt_size, shuffle=True)

        # -------------------------- build models --------------------------#
        from models.vit import vit_base_patch16_224
        model = vit_base_patch16_224(num_classes=2).cuda()
        print(model)
        pretrained = True
        if pretrained:#这一段要如何修改
            model_dict = model.state_dict()
            model_weights = torch.load(parse_config.weight_path)
            pretrained_dict = model_weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#取出预训练模型中与新模型的dict中重合的部分
            model_dict.update(pretrained_dict)#用预训练模型参数更新new_model中的部分参数
            model.load_state_dict(model_dict) #将更新后的model_dict加载进new model中 

        cls_loss2 = nn.CrossEntropyLoss()
        # -------------------------- start training --------------------------#
        max_iou = 0
        best_ep = 0
        min_loss = 10
        min_epoch = 0
        # -------------------------- build loggers and savers --------------------------#
        os.makedirs(parse_config.results_dir, exist_ok=True)
        writer = SummaryWriter(parse_config.results_dir)
        log_path = parse_config.results_dir
        EPOCHS = parse_config.n_epochs
        logging.basicConfig(filename=os.path.join(log_path,'train_log.log'),
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        log_path_foldTP1 = log_path + 'fold{}/TP1'.format(fold)
        os.makedirs(log_path_foldTP1, exist_ok=True)
        log_path_foldLOW = log_path + 'fold{}/LOW'.format(fold)
        os.makedirs(log_path_foldLOW, exist_ok=True)
        log_path_foldFN = log_path + 'fold{}/FN'.format(fold)
        os.makedirs(log_path_foldFN, exist_ok=True)
        log_path_foldFP = log_path + 'fold{}/FP'.format(fold)
        os.makedirs(log_path_foldFP, exist_ok=True)
        log_path_foldTN = log_path + 'fold{}/TN'.format(fold)
        os.makedirs(log_path_foldTN, exist_ok=True)
        csv_file_score = log_path + 'fold{}/'.format(fold) + 'score.csv'
        csv_file_score_Hight = log_path + 'fold{}/'.format(fold) + 'scoreHight.csv'
        csv_file_score_LOW = log_path + 'fold{}/'.format(fold) + 'scoreLOW.csv'


        # start training
        for epoch in range(1, EPOCHS + 1):
            #print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            start = time.time()
            eval_metrics, nameFP, nameFN, nameTP1, nameTP2, nameTN, name_low, subscore1, subscore2, allscore= test(epoch, model, loader_eval, cls_loss2, fold)
            time_elapsed = time.time() - start

            print()
            print(
                'Training on epoch:{} complete in {:.0f}m {:.0f}s'.
                    format(epoch, time_elapsed // 60, time_elapsed % 60))
            print('valid/loss', eval_metrics['loss'])
            print('valid/acc',  eval_metrics['acc'])
            print('valid/f1', eval_metrics['f1'])
            print('valid/kappa', eval_metrics['kappa'])
            print('valid/specificity', eval_metrics['specificity'])
            print('valid/recall', eval_metrics['recall'])
            print('valid/precision', eval_metrics['precision'])
            print('confusion_matrix:',eval_metrics['confusion_matrix'])
            print()
            logging.info(
                '\n Training on epoch:{} complete in {:.0f}m {:.0f}s'.
                    format(epoch, time_elapsed // 60, 
                        time_elapsed % 60))
            logging.info(
                ' \n valid/loss:{} \n valid/acc:{} \n valid/f1:{} \n valid/kappa:{} \n valid/recall:{} \n valid/precision:{} \n valid/specificity:{} \n confusion_matrix:{} \n '.
                    format(
                        round(eval_metrics['loss'], 3), 
                        round(eval_metrics['acc'], 3), 
                        round(eval_metrics['f1'], 3), 
                        round(eval_metrics['kappa'], 3),
                        round(eval_metrics['recall'], 3),
                        round(eval_metrics['precision'], 3),
                        round(eval_metrics['specificity'], 3),
                        eval_metrics['confusion_matrix']))
            
            
            # 遍历 correct_preds1 列表，复制对应的文件到目标目录
            for filename in nameTP1:
                if filename[0].endswith('.npz'):
                    source_file = os.path.join(parse_config.data_dir + '/png1', filename[0].replace('.npz', '.png'))
                    target_file = os.path.join(log_path_foldTP1, filename[0].replace('.npz', '.png'))
                    shutil.copy2(source_file, target_file)
                    with open(log_path_foldTP1 + '/file.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([filename[0]])

            for filename in name_low:
                if filename[0].endswith('.npz'):
                    source_file = os.path.join(parse_config.data_dir + '/png1', filename[0].replace('.npz', '.png'))
                    target_file = os.path.join(log_path_foldLOW, filename[0].replace('.npz', '.png'))
                    shutil.copy2(source_file, target_file)
                    with open(log_path_foldLOW + '/file.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([filename[0]])

            for filename in nameFN:
                if filename[0].endswith('.npz'):
                    source_file = os.path.join(parse_config.data_dir + '/png1', filename[0].replace('.npz', '.png'))
                    target_file = os.path.join(log_path_foldFN, filename[0].replace('.npz', '.png'))
                    shutil.copy2(source_file, target_file)
                    with open(log_path_foldFN + '/file.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([filename[0]])

            for filename in nameFP:
                if filename[0].endswith('.npz'):
                    source_file = os.path.join(parse_config.data_dir + '/png1', filename[0].replace('.npz', '.png'))
                    target_file = os.path.join(log_path_foldFP, filename[0].replace('.npz', '.png'))
                    shutil.copy2(source_file, target_file)
                    with open(log_path_foldFP + '/file.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([filename[0]])

            for filename in nameTN:
                if filename[0].endswith('.npz'):
                    source_file = os.path.join(parse_config.data_dir + '/png1', filename[0].replace('.npz', '.png'))
                    target_file = os.path.join(log_path_foldTN, filename[0].replace('.npz', '.png'))
                    shutil.copy2(source_file, target_file)
                    with open(log_path_foldTN + '/file.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([filename[0]])

        with open(csv_file_score, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'label', 'score', 'pre'])
            for item in allscore:
                writer.writerow([item['name'], item['label'], item['score'], item['pre']])   

        with open(csv_file_score_Hight, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'label', 'score', 'pre'])
            for item in subscore1:
                writer.writerow([item['name'], item['label'], item['score'], item['pre']])      

        with open(csv_file_score_LOW, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'label', 'score', 'pre'])
            for item in subscore2:
                writer.writerow([item['name'], item['label'], item['score'], item['pre']])   

        global_nameFN.append(nameFN)
        global_nameFP.append(nameFP)
        global_nameTP1.append(nameTP1)
        global_nameLOW.append(name_low)
        global_nameTN.append(nameTN)
        global_accuracy.append(round(eval_metrics['acc'], 4))
        global_sensitivity.append(round(eval_metrics['recall'], 4))
        global_F1_Score.append(round(eval_metrics['f1'], 4))

    c95_sensitivity = get_ci(global_sensitivity) 
    c95_accuracy = get_ci(global_accuracy) 
    c95_F1_Score = get_ci(global_F1_Score) 

    logging.info(
    '\n global_recall:{} \n  global_accuracy:{} \n global_F1_Score:{} '.
        format(global_sensitivity, global_accuracy, global_F1_Score))
    logging.info(
    '\n c95_recall:{} \n c95_accuracy:{} \n c95_F1_Score:{}'.
        format(c95_sensitivity, c95_accuracy, c95_F1_Score))

    #----------------将预测错误的保存到 ./logs_val验证/FN.txt -------------------------------------
    with open(os.path.join(parse_config.results_dir, 'FN.txt'), 'w') as file:
        for item in global_nameFN:
            for item_sub in item:
                file.write(f'{item_sub}\n')
    # ------------------------------------------------------------------------------------------
        
    #----------------将预测错误的保存到 ./logs_val验证/FP.txt -------------------------------------
    with open(os.path.join(parse_config.results_dir, 'FP.txt'), 'w') as file:
        for item in global_nameFP:
            for item_sub in item:
                file.write(f'{item_sub}\n')
    # -------------------------------------------------------------------------------------------

    #----------------将预测正确的1保存到 ./logs_val验证/TP1.txt -------------------------------------
    with open(os.path.join(parse_config.results_dir, 'TP1.txt'), 'w') as file:
        for item in global_nameTP1:
            for item_sub in item:
                file.write(f'{item_sub}\n')

    with open(os.path.join(parse_config.results_dir, 'TP2.txt'), 'w') as file:
        for item in global_nameLOW:
            for item_sub in item:
                file.write(f'{item_sub}\n')
    # --------------------------------------------------------------------------------------------
                
    #----------------将预测错误的保存到 ./logs_val验证/TN.txt -------------------------------------
    with open(os.path.join(parse_config.results_dir, 'TN.txt'), 'w') as file:
        for item in global_nameTN:
            for item_sub in item:
                file.write(f'{item_sub}\n')
    # ------------------------------------------------------------------------------------------