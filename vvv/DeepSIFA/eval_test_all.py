import os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import numpy as np
from tqdm import tqdm
import sys
import gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.utils.data
from scipy import stats
from utils.xrayloader import XrayDataset_val
from utils.metrics_2cls import *
from collections import OrderedDict
import pandas as pd
from PIL import Image
import shutil
import time
import csv
import glob



def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1)  
    parser.add_argument('--bt_size', type=int, default=1)
    parser.add_argument('--weight_path', type=str, default=os.path.join('DeepSIFA', 'checkpoints', 'best_acc.pth'))
    parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'wt_0119', 'I'))
    parser.add_argument('--results_dir', default=os.path.join('DeepSIFA'), type=str, metavar='FILENAME',
                        help='Output csv file for validation results (summary)')
    parse_config = parser.parse_args()
    return parse_config


def compute_metrics(outputs, targets, loss_fn):
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()
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
def test(parse_config, epoch, model, loader_eval, loss_fn):
    print("-------------testing-----------")
    model.eval()
    predictions = []
    allscore = []
    labels = []

    with torch.no_grad():
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
                'score':round(realprobs.item(), 2),
                'pre': max_indices.item()
            }
            allscore.append(result_dict) 
            predictions.append(output)
            labels.append(label_onehot)



    return allscore


def get_ci(global_list):
    average_value = round(np.mean(global_list), 3)
    standard_error  = stats.sem(global_list)
    a = round(average_value - 1.96 * standard_error, 3)
    b = round(average_value + 1.96 * standard_error, 3)
    return [average_value, (a, b)]


if __name__ == '__main__':
    parse_config = get_cfg()
    folder_path = parse_config.data_dir
    # 遍历文件夹中的所有 .tif 文件
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    for idx, filename in enumerate(tif_files):
        file_path = os.path.join(folder_path, filename)
        TIF_NAME = filename.replace('.tif', '')
        print('选取的TIRF图片为',TIF_NAME)

        global_nameFN = []
        global_nameFP = []
        global_nameTP = []
        global_nameLOW = []
        global_nameHIGH = []
        global_nameTN = []
        global_accuracy = []
        global_sensitivity = [] 
        global_F1_Score = []
        # -------------------------- get args --------------------------#
        gc.collect()
        torch.cuda.empty_cache()
        parse_config = get_cfg()
        print('data_dir_TIF_NAME路径', os.path.join(parse_config.data_dir, TIF_NAME))
        parse_config.data_dir1 = os.path.join(parse_config.data_dir, TIF_NAME, 'processing_data')
        parse_config.val_csv_dir = glob.glob(os.path.join(parse_config.data_dir, TIF_NAME, '*.csv'))[0]


        # -------------------------- build dataloaders --------------------------#
        dataset = XrayDataset_val(parse_config)
        loader_eval = torch.utils.data.DataLoader(dataset, batch_size=parse_config.bt_size, shuffle=True)

        # -------------------------- build models --------------------------#
        from models.vit import vit_base_patch16_224
        model = vit_base_patch16_224(num_classes=2).cuda()
        # print(model)
        pretrained = True
        if pretrained:#这一段要如何修改
            model_dict = model.state_dict()
            model_weights = torch.load(parse_config.weight_path, weights_only=True)
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
        directory = os.path.join(parse_config.results_dir, 'logs_test_{}'.format(TIF_NAME))
        if os.path.exists(directory):
            shutil.rmtree(directory)  # 删除整个目录
        os.makedirs(directory, exist_ok=True)  # 重新创建目录
        EPOCHS = parse_config.n_epochs

        
        csv_file_score = os.path.join(directory, 'score.csv')
        csv_file_score_HIGH = os.path.join(directory, 'scoreHIGH.csv')
        csv_file_score_LOW = os.path.join(directory, 'scoreLOW.csv')
        csv_file_score_MIDDLE = os.path.join(directory, 'scoreMIDDLE.csv')



        # --------------------------------start training------------------------------
        for epoch in range(1, EPOCHS + 1):
            #print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            start = time.time()
            allscore= test(parse_config, epoch, model, loader_eval, cls_loss2)
            time_elapsed = time.time() - start

        with open(csv_file_score, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'label', 'score', 'pre'])
            for item in allscore:
                writer.writerow([item['name'], item['label'], item['score'], item['pre']])   



        # # -------------------------------------------画概率图 全部-------------------------------------------
        # 读取CSV文件
        df = pd.read_csv(csv_file_score)
        scores = df['score']
        bin_sizes = [0.025, 0.0125, 0.00625]
        image_files = []  # 存储生成的图片路径

        for bin_size in bin_sizes:
            bins = np.arange(0, 1 + bin_size, bin_size)
            output_file = os.path.join(directory, f'score_distribution_{bin_size:.4f}.png')
            image_files.append(output_file)  # 记录图片路径
            
            plt.figure(figsize=(10, 6))
            counts, bins, patches = plt.hist(scores, bins=bins, edgecolor='black', alpha=0.7)
            total = len(scores)
            plt.title(f'Score Distribution (Total: {total}, Bin Size: {bin_size:.4f})')
            plt.xlabel('Score', labelpad=0)
            plt.ylabel('Num')
            plt.grid(axis='y', alpha=0.75)
            plt.xticks(np.arange(0, 1.1, 0.1))

            # 动态调整字体大小
            fontsize = 2 if bin_size == 0.00625 else (3 if bin_size == 0.0125 else 4)
            for count, bin_edge in zip(counts, bins[:-1]):
                if count > 0:
                    percentage = f'{(count / total * 100):.1f}%'
                    plt.text(bin_edge + bin_size / 2, count, percentage, ha='center', va='bottom', fontsize=fontsize, color='black')

            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.07)
            plt.savefig(output_file, dpi=300)
            plt.close()
            # print(f"柱状图已保存到 {output_file}")

        # 合并生成的图片
        images = [Image.open(image_file) for image_file in image_files]
        total_height = sum(image.height for image in images)
        max_width = max(image.width for image in images)

        # 创建一个新的图像来存放拼接后的结果
        merged_image = Image.new('RGB', (max_width, total_height))

        # 在新图像中粘贴每个图像
        y_offset = 0
        for image in images:
            merged_image.paste(image, (0, y_offset))
            y_offset += image.height

        # 保存拼接后的图像
        merged_output_file = os.path.join(directory, 'score_distribution.png')
        merged_image.save(merged_output_file)

        print(f"score_distribution拼接后的图片已保存到 {merged_output_file}")
        # 删除之前生成的三个子图
        for image_file in image_files:
            os.remove(image_file)
            # print(f"已删除生成的子图: {image_file}")
        print(f"模型预测 完成")






