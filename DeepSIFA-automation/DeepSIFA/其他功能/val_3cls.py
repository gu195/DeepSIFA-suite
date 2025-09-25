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

from torch.optim.lr_scheduler import CosineAnnealingLR
import time

from utils.xrayloader import XrayDataset_val_mulcls
from utils.metrics_4cls import *
from collections import OrderedDict
import statistics
OUT_FILENAME = 'xmu1+zs_3cls_results.txt'
from scipy import stats

def get_cfg(fold):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default='512', type=int)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float, default=1e-4)  # 0.0003
    parser.add_argument('--n_epochs', type=int, default=1)  
    parser.add_argument('--bt_size', type=int, default=1)  # 36

    # github上路径
    parser.add_argument('--weight_path', type=str, default='./checkpoints/fold{}/best_acc.pth'.format(fold))
    parser.add_argument('--data_dir', type=str, default='../data_1/xmu1/图片v2/segmentation_lables/images')
    parser.add_argument('--train_csv_dir', type=str, default='../data_1/xmu1/5折交叉验证_42/train_fold{}_多标签.csv'.format(fold))
    parser.add_argument('--val_csv_dir', type=str, default='../data_1/xmu1/5折交叉验证_42/val_fold{}_多标签.csv'.format(fold))
    parser.add_argument('--results_dir', default='./验证集的多分类结果01/fold{}/'.format(fold), type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')

    parse_config = parser.parse_args()
    return parse_config

def compute_metrics(outputs, targets, roc_name, args):
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    precision = Precision(outputs, targets)
    ppv = PPV(outputs, targets)
    npv = NPV(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    roc(outputs, targets, args.results_dir, roc_name)
    specificity = spe(outputs, targets)
    metrics = OrderedDict([
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

def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def structure_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_score(y_true, y_pred):
    """
    y_true:(b,c,h,w) label
    y_pred:(b,c,h,w) prediction
    按照每个样本计算
    """
    return ((2 * (y_true * y_pred).sum(-1).sum(-1).sum(-1) + 1e-15) / (y_true.sum(-1).sum(-1).sum(-1) + y_pred.sum(-1).sum(-1).sum(-1) + 1e-15)).mean()

def normalize(img):
    return (img-img.min()) / (img.max()-img.min())

def get_ci(global_list):
    average_value = round(np.mean(global_list), 3)
    standard_error  = stats.sem(global_list)
    a = round(average_value - 1.96 * standard_error, 3)
    b = round(average_value + 1.96 * standard_error, 3)
    return [average_value, (a, b)]

# -------------------------- test func --------------------------#
def test(epoch, model, loader_eval, args):
    print("-------------testing-----------")
    model.eval()
    accuracy_4keys = 0
    total_correct = 0
    total_samples = 0
    accuracy_4keys = 0
    total_correct = 0
    total_samples = 0
    labels_1 = []
    labels_2 = []
    labels_3 = []
    predicted_key1 = []
    predicted_key2 = []
    predicted_key3 = []
    previous_labels = set()  # 用于存储之前已经记录的label_4keys
    label_counts = {}  # 用字典来记录label_4keys和对应的出现次数

    model.eval()
    with torch.no_grad():
        for input, _, label1, label2, label3, labelcls, name in tqdm(loader_eval):
            label1 = label1
            input = input.cuda()
            _, _, task1_output, task2_output, task3_output, task4_output = model(input)

            labels_1.append(label1)
            labels_2.append(label2)
            labels_3.append(label3)

            predicted_key1.append(task1_output.argmax(1))
            predicted_key2.append(task2_output.argmax(1))
            predicted_key3.append(task3_output.argmax(1))
            predicted_label = task4_output.argmax(1).unsqueeze(0)
            predicted = torch.cat([task1_output.argmax(1).unsqueeze(0), task2_output.argmax(1).unsqueeze(0), task3_output.argmax(1).unsqueeze(0), predicted_label], dim=1).cpu()
            label_4keys = torch.cat([label1, label2, label3, labelcls], dim=0).unsqueeze(0)
            total_samples += labelcls.size(0)

            if (predicted == label_4keys).sum().item() == 4:
                total_correct += 1
                label_4keys_tolist = label_4keys.tolist()
                label_4keys_flat = label_4keys_tolist[0]  # 获取第一个子列表
                label_4keys_tuple = tuple(label_4keys_flat)  # 转换为元组
                if not any(label_4keys_tuple == label for label in previous_labels):
                    logging.info(
                        '\n 第{}个: \n 4-keys full correctly:{} \n label_keys:{} \n predicted1:{} \n'.
                            format(total_correct, name, label_4keys_tuple, predicted))
                previous_labels.add(label_4keys_tuple)  # 将新的label_4keys_tuple加入到记录的集合中

                # 统计label_4keys出现次数
                if label_4keys_tuple in label_counts:
                    label_counts[label_4keys_tuple] += 1
                else:
                    label_counts[label_4keys_tuple] = 1

    # 打印每个label_4keys的出现次数
    for label, count in label_counts.items():
        logging.info(f'Label: {label}, 出现次数: {count}')
    accuracy_4keys = 100 * total_correct / total_samples
    evaluation_metrics_1 = compute_metrics(predicted_key1, labels_1, 'ROC_1bone', args)
    evaluation_metrics_2 = compute_metrics(predicted_key2, labels_2, 'ROC_2bone_trabecula', args)
    evaluation_metrics_3 = compute_metrics(predicted_key3, labels_3, 'ROC_3fracture_line', args)
    print("accuracy of 4keys = ", accuracy_4keys)
    logging.info('accuracy of 4keys:{}\n'.format(accuracy_4keys))

    return evaluation_metrics_1, evaluation_metrics_2, evaluation_metrics_3, accuracy_4keys

def write_results2txt(results_dir, results, name):
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, name + OUT_FILENAME)
    file = open(results_file, 'w')
    file.write(results)
    file.close()

if __name__ == '__main__':
    global_sensitivity_1 = []  
    global_specificity_1 = []
    global_accuracy_1 = []
    global_F1_Score_1 = []
    global_PPV_1 = []
    global_NPV_1 = []

    global_sensitivity_2 = []  
    global_specificity_2 = []
    global_accuracy_2 = []
    global_F1_Score_2 = []
    global_PPV_2 = []
    global_NPV_2 = []

    global_sensitivity_3 = []  
    global_specificity_3 = []
    global_accuracy_3 = []
    global_F1_Score_3 = []
    global_PPV_3 = []
    global_NPV_3 = []

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
        dataset = XrayDataset_val_mulcls(parse_config, transform)
        loader_eval = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        # -------------------------- build models --------------------------#
        from models.resnet import resnet34
        from models.deeplabv3plus import DeepLabHeadV3Plus, SegmentationModel
        bkbone = resnet34().cuda()
        head = DeepLabHeadV3Plus(in_channels=512, low_level_channels=64, num_classes=1)
        model = SegmentationModel(bkbone, head).cuda()
        if fold == 1:
            print(model)
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
        os.makedirs(parse_config.results_dir)
        writer = SummaryWriter(parse_config.results_dir)
        log_path = parse_config.results_dir
        EPOCHS = parse_config.n_epochs
        logging.basicConfig(filename=os.path.join(log_path,'train_log_3cls.log'),
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

        # start training
        for epoch in range(1, EPOCHS + 1):
            #print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            start = time.time()
            results1, results2, results3, accuracy_4keys  = test(epoch, model, loader_eval, args=parse_config)
            time_elapsed = time.time() - start

            print()
            print(
                'Training on epoch:{} complete in {:.0f}m {:.0f}s'.
                    format(epoch, time_elapsed // 60, time_elapsed % 60))
            print('骨质皮质valid/acc',  results1['acc'])
            print('骨质皮质valid/f1', results1['f1'])
            print('骨质皮质valid/kappa', results1['kappa'])
            print('骨质皮质valid/specificity', results1['specificity'])
            print('骨质皮质valid/recall', results1['recall'])
            print('骨质皮质valid/ppv', results1['ppv'])
            print('骨质皮质valid/npv', results1['npv'])
            print('骨质皮质confusion_matrix:',results1['confusion_matrix'])
            print()
            logging.info(
                ' \n 第{}折: \n 骨质皮质 \n valid/acc:{} \n valid/f1:{} \n valid/kappa:{} \n valid/recall:{} \n valid/specificity:{} \n valid/ppv:{} \n valid/npv:{} \n confusion_matrix:{} \n'.
                    format(
                        fold,
                        round(results1['acc'], 3),
                        round(results1['f1'], 3),
                        round(results1['kappa'], 3),
                        round(results1['recall'], 3),
                        round(results1['specificity'], 3),
                        round(results1['ppv'], 3),
                        round(results1['npv'], 3),
                        results1['confusion_matrix']))
            logging.info(
                ' \n 第{}折: \n 骨小梁 \n valid/acc:{} \n valid/f1:{} \n valid/kappa:{} \n valid/recall:{} \n valid/specificity:{} \n valid/ppv:{} \n valid/npv:{} \n confusion_matrix:{} \n'.
                    format(
                        fold,
                        round(results2['acc'], 3),
                        round(results2['f1'], 3),
                        round(results2['kappa'], 3),
                        round(results2['recall'], 3),
                        round(results2['specificity'], 3),
                        round(results2['ppv'], 3),
                        round(results2['npv'], 3),
                        results2['confusion_matrix']))
            logging.info(
                ' \n 第{}折: \n 骨折线 \n valid/acc:{} \n valid/f1:{} \n valid/kappa:{} \n valid/recall:{} \n valid/specificity:{} \n valid/ppv:{} \n valid/npv:{} \n confusion_matrix:{} \n'.
                    format(
                        fold,
                        round(results3['acc'], 3),
                        round(results3['f1'], 3),
                        round(results3['kappa'], 3),
                        round(results3['recall'], 3),
                        round(results3['specificity'], 3),
                        round(results3['ppv'], 3),
                        round(results3['npv'], 3),
                        results3['confusion_matrix']))
            
        
        global_sensitivity_1.append(round(results1['recall'], 4))
        global_specificity_1.append(round(results1['specificity'], 4))
        global_accuracy_1.append(round(results1['acc'], 4))
        global_F1_Score_1.append(round(results1['f1'], 4))
        global_PPV_1.append(round(results1['ppv'], 4))
        global_NPV_1.append(round(results1['npv'], 4))

        global_sensitivity_2.append(round(results2['recall'], 4))
        global_specificity_2.append(round(results2['specificity'], 4))
        global_accuracy_2.append(round(results2['acc'], 4))
        global_F1_Score_2.append(round(results2['f1'], 4))
        global_PPV_2.append(round(results2['ppv'], 4))
        global_NPV_2.append(round(results2['npv'], 4))

        global_sensitivity_3.append(round(results3['recall'], 4))
        global_specificity_3.append(round(results3['specificity'], 4))
        global_accuracy_3.append(round(results3['acc'], 4))
        global_F1_Score_3.append(round(results3['f1'], 4))
        global_PPV_3.append(round(results3['ppv'], 4))
        global_NPV_3.append(round(results3['npv'], 4))
        
    c95_sensitivity_1 = get_ci(global_sensitivity_1) 
    c95_specificity_1= get_ci(global_specificity_1) 
    c95_accuracy_1 = get_ci(global_accuracy_1) 
    c95_F1_Score_1 = get_ci(global_F1_Score_1) 
    c95_PPV_1 = get_ci(global_PPV_1) 
    c95_NPV_1 = get_ci(global_NPV_1) 

    c95_sensitivity_2 = get_ci(global_sensitivity_2) 
    c95_specificity_2= get_ci(global_specificity_2) 
    c95_accuracy_2 = get_ci(global_accuracy_2) 
    c95_F1_Score_2 = get_ci(global_F1_Score_2) 
    c95_PPV_2 = get_ci(global_PPV_2) 
    c95_NPV_2 = get_ci(global_NPV_2) 

    c95_sensitivity_3 = get_ci(global_sensitivity_3) 
    c95_specificity_3= get_ci(global_specificity_3) 
    c95_accuracy_3 = get_ci(global_accuracy_3) 
    c95_F1_Score_3 = get_ci(global_F1_Score_3) 
    c95_PPV_3 = get_ci(global_PPV_3) 
    c95_NPV_3 = get_ci(global_NPV_3) 


    logging.info(
    '\n 骨质皮质 \n global_sensitivity:{} \n global_specificity:{} \n global_accuracy:{} \n global_F1_Score:{} \n global_PPV:{} \n global_NPV:{} \n'.
        format(global_sensitivity_1, global_specificity_1, global_accuracy_1, global_F1_Score_1, global_PPV_1, global_NPV_1))
    logging.info(
    '\n 骨质皮质 \n c95_sensitivity:{} \n c95_specificity:{} \n c95_accuracy:{} \n c95_F1_Score:{} \n c95_PPV:{} \n c95_NPV:{} \n'.
        format(c95_sensitivity_1, c95_specificity_1, c95_accuracy_1, c95_F1_Score_1, c95_PPV_1, c95_NPV_1))
    
    logging.info(
    '\n 骨小梁 \n global_sensitivity:{} \n global_specificity:{} \n global_accuracy:{} \n global_F1_Score:{} \n global_PPV:{} \n global_NPV:{} \n'.
        format(global_sensitivity_2, global_specificity_2, global_accuracy_2, global_F1_Score_2, global_PPV_2, global_NPV_2))
    logging.info(
    '\n  骨小梁 \n c95_sensitivity:{} \n c95_specificity:{} \n c95_accuracy:{} \n c95_F1_Score:{} \n c95_PPV:{} \n c95_NPV:{} \n'.
        format(c95_sensitivity_2, c95_specificity_2, c95_accuracy_2, c95_F1_Score_2, c95_PPV_2, c95_NPV_2))
    
    logging.info(
    '\n 骨折线 \n global_sensitivity:{} \n global_specificity:{} \n global_accuracy:{} \n global_F1_Score:{} \n global_PPV:{} \n global_NPV:{} \n'.
        format(global_sensitivity_3, global_specificity_3, global_accuracy_3, global_F1_Score_3, global_PPV_3, global_NPV_3))
    logging.info(
    '\n 骨折线 \n c95_sensitivity:{} \n c95_specificity:{} \n c95_accuracy:{} \n c95_F1_Score:{} \n c95_PPV:{} \n c95_NPV:{} \n'.
        format(c95_sensitivity_3, c95_specificity_3, c95_accuracy_3, c95_F1_Score_3, c95_PPV_3, c95_NPV_3))
    
    # output_str1 = '皮质骨质中断 Test Results:\n'
    # for key, value in results1.items():
    #     if key == 'confusion matrix':
    #         output_str1 += f'{key}:\n {value}\n'
    #     elif key == 'classification report':
    #         output_str1 += f'{key}:\n {value}\n'
    #     else:
    #         output_str1 += f'{key}: {value}\n'
    # write_results2txt(parse_config.results_dir, output_str1, '皮质骨质中断')

    # output_str2 = '骨小梁中断 Test Results:\n'
    # for key, value in results2.items():
    #     if key == 'confusion matrix':
    #         output_str2 += f'{key}:\n {value}\n'
    #     elif key == 'classification report':
    #         output_str2 += f'{key}:\n {value}\n'
    #     else:
    #         output_str2 += f'{key}: {value}\n'
    # write_results2txt(parse_config.results_dir, output_str2, '骨小梁中断紊乱')

    # output_str3 = '可见骨折线 Test Results:\n'
    # for key, value in results3.items():
    #     if key == 'confusion matrix':
    #         output_str3 += f'{key}:\n {value}\n'
    #     elif key == 'classification report':
    #         output_str3 += f'{key}:\n {value}\n'
    #     else:
    #         output_str3 += f'{key}: {value}\n'
    # write_results2txt(parse_config.results_dir, output_str3, '可见骨折线')




