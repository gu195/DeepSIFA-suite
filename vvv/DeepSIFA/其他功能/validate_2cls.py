import os, argparse, math
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
import gc
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import time
from utils.xrayloader import XrayDataset_2cls_jlu2
from utils.metrics_2cls import *
from collections import OrderedDict
from scipy import stats
import csv
import cv2
GLOBALQ = 25
VERSION = 25

def get_cfg(fold):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default='512', type=int)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float, default=1e-4)  # 0.0003
    parser.add_argument('--n_epochs', type=int, default=1)  
    parser.add_argument('--bt_size', type=int, default=1)  # 36

    # github上路径
    parser.add_argument('--weight_path', type=str, default='./checkpoints/fold{}/best_acc.pth'.format(fold))
    # parser.add_argument('--weight_path', type=str, default='./checkpoints/best_accuracy_4keys.pth')
    parser.add_argument('--data_dir', type=str, default='../data_7/jl2/图片_q{}/segmentation_lables/images'.format(GLOBALQ))
    parser.add_argument('--csv_dir', type=str, default='../data_7/jl2/label344_.csv'.format(fold))
    parser.add_argument('--results_dir', default='./log_jlu2外部验证_q{}/fold{}/'.format(GLOBALQ, fold), type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')

    parse_config = parser.parse_args()
    return parse_config

def compute_metrics(outputs, targets, loss_fn):
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()
    loss = loss_fn(outputs, targets).cpu().item()
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    precision = Precision(outputs, targets)
    ppv = PPV(outputs, targets)
    npv = NPV(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    specificity = spe(outputs, targets)
    metrics = OrderedDict([
        ('loss', loss),
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('ppv', ppv),
        ('npv', npv),
        ('kappa', kappa),
        ('confusion_matrix',cm),
        ('specificity',specificity)
    ])
        
    return metrics

def imshow(img, out_dir, name):
    # img 是model的输出mask 512 512
    for i in range(img.shape[0]):
        npimg = img.cpu().numpy()
        out_pre = out_dir + '/pre/'
        os.makedirs(out_pre, exist_ok=True)
        cv2.imwrite(out_pre + name[i], npimg[i][0]*255)
    return

def get_ci(global_list):
    average_value = round(np.mean(global_list), 3)
    standard_error  = stats.sem(global_list)
    a = round(average_value - 1.96 * standard_error, 3)
    b = round(average_value + 1.96 * standard_error, 3)
    return [average_value, (a, b)]

# -------------------------- test func --------------------------#
def test(epoch, model, out_dir, loader_eval, loss_fn):
    print("-------------testing-----------")
    model.eval()
    predictions = []
    labels = []
    wrong_preds = []
    wrong_preds_name = []

    with torch.no_grad():
        # 初始化错误预测列表
        wrong_preds = [['name label pre']]
        for img, label, name in tqdm(loader_eval):
            img = img.cuda().float()
            label = label.cuda()
            pre, x_512, task1_output, task2_output, task3_output, output = model(img)
            pre = pre>0.5
            imshow(pre, out_dir, name)

            if isinstance(output, (tuple, list)):
                output = output[0]

            predictions.append(output)
            labels.append(label)

            preds = torch.argmax(output, dim=1)
            # 记录预测错误的文件名
            if preds != label:
                name_message = [name[0], label.item(), preds.item()]
                wrong_preds.append(name_message)
                wrong_preds_name.append(name[0])
                print('Incorrect prediction: {}'.format(name))

    with open('./log_jlu2外部验证_q{}/fold{}/jlu2_wrong_predictions.txt'.format(GLOBALQ, fold), 'w') as file:
        for item in wrong_preds:
                file.write(f'{item}\n')
    evaluation_metrics = compute_metrics(predictions, labels, loss_fn)
    return evaluation_metrics, wrong_preds, wrong_preds_name


if __name__ == '__main__':
    global_sensitivity = []  
    global_specificity = []
    global_accuracy = []
    global_F1_Score = []
    global_PPV = []
    global_NPV = []
    global_wrong_preds = []
    global_wrong_preds_name = []
    for fold in range(5,6):
        print('第{}折:'.format(fold))
        # -------------------------- get args --------------------------#
        gc.collect()
        torch.cuda.empty_cache()
        parse_config = get_cfg(fold)
        # -------------------------- build dataloaders --------------------------#

        transform = transforms.Compose([
        transforms.Resize((parse_config.img_size, parse_config.img_size)),
        transforms.ToTensor()
        ])
        dataset = XrayDataset_2cls_jlu2(parse_config, transform)
        loader_eval = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        # -------------------------- build models --------------------------#
        from models.resnet import resnet34
        from models.deeplabv3plus import DeepLabHeadV3Plus, SegmentationModel
        bkbone = resnet34().cuda()
        head = DeepLabHeadV3Plus(in_channels=512, low_level_channels=64, num_classes=1)
        model = SegmentationModel(bkbone, head).cuda()
        # print(model)
        pretrained = True
        if pretrained:#这一段要如何修改
            model_dict = model.state_dict()
            model_weights = torch.load(parse_config.weight_path.format(fold))
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
        os.makedirs(parse_config.results_dir ,exist_ok=True)
        writer = SummaryWriter(parse_config.results_dir)
        log_path = parse_config.results_dir
        EPOCHS = parse_config.n_epochs
        logging.basicConfig(filename=os.path.join(log_path,'train_log_q{}.log'.format(GLOBALQ)),
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        out_images_path = parse_config.results_dir + 'val_images'
        os.makedirs(out_images_path, exist_ok=True)

        # start training
        for epoch in range(1, EPOCHS + 1):
            #print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            start = time.time()
            eval_metrics, wrong_preds, wrong_preds_name = test(epoch, model, out_images_path, loader_eval, cls_loss2)

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
            print('valid/ppv', eval_metrics['ppv'])
            print('valid/npv', eval_metrics['npv'])
            print('confusion_matrix:',eval_metrics['confusion_matrix'])
            print()
            logging.info(
                ' \n 第{}折: \n valid/loss:{} \n valid/acc:{} \n valid/f1:{} \n valid/kappa:{} \n valid/recall:{} \n valid/specificity:{} \n valid/ppv:{} \n valid/npv:{} \n confusion_matrix:{} \n'.
                    format(
                        fold,
                        eval_metrics['loss'], 
                        eval_metrics['acc'], 
                        eval_metrics['f1'], 
                        eval_metrics['kappa'],
                        eval_metrics['recall'],
                        eval_metrics['specificity'],
                        eval_metrics['ppv'],
                        eval_metrics['npv'],
                        eval_metrics['confusion_matrix']))
            
        
        global_sensitivity.append(round(eval_metrics['recall'], 4))
        global_specificity.append(round(eval_metrics['specificity'], 4))
        global_accuracy.append(round(eval_metrics['acc'], 4))
        global_F1_Score.append(round(eval_metrics['f1'], 4))
        global_PPV.append(round(eval_metrics['ppv'], 4))
        global_NPV.append(round(eval_metrics['npv'], 4))
        global_wrong_preds.append(wrong_preds)
        global_wrong_preds_name.append(wrong_preds_name)
        

    c95_sensitivity = get_ci(global_sensitivity) 
    c95_specificity = get_ci(global_specificity) 
    c95_accuracy = get_ci(global_accuracy) 
    c95_F1_Score = get_ci(global_F1_Score) 
    c95_PPV = get_ci(global_PPV) 
    c95_NPV = get_ci(global_NPV) 

    output_directory = './log_jlu2外部验证_q25/logs_jl2_错误文件名/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_path1 = os.path.join(output_directory, 'wrong_preds_v{}.txt'.format(VERSION))
    with open(output_file_path1, 'w') as file:
        for item in global_wrong_preds:
            for item_sub in item:
                file.write(f'{item_sub}\n')
    print("Wrong predictions saved to wrong_preds.txt")

    output_file_path2 = os.path.join(output_directory, 'wrong_predsname_v{}.csv'.format(VERSION))
    with open(output_file_path2, 'w', newline='') as csvfile:  # 使用newline=''以确保换行符在CSV中正常处理
        writer = csv.writer(csvfile)
        writer.writerow(['file_name'])
        for item in global_wrong_preds_name:
            for item_sub in item:
                writer.writerow([item_sub])
    print("Wrong predictions saved to wrong_preds_v{}.csv".format(VERSION))

    logging.info(
    '\n global_sensitivity:{} \n global_specificity:{} \n global_accuracy:{} \n global_F1_Score:{} \n global_PPV:{} \n global_NPV:{} \n'.
        format(global_sensitivity, global_specificity, global_accuracy, global_F1_Score, global_PPV, global_NPV))
    logging.info(
    '\n c95_sensitivity:{} \n c95_specificity:{} \n c95_accuracy:{} \n c95_F1_Score:{} \n c95_PPV:{} \n c95_NPV:{} \n'.
        format(c95_sensitivity, c95_specificity, c95_accuracy, c95_F1_Score, c95_PPV, c95_NPV))




