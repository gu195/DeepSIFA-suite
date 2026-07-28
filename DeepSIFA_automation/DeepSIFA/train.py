import os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
from tqdm import tqdm
import sys
import gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.xrayloader import XrayDataset_train, XrayDataset_val
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer_v2, optimizer_kwargs
from utils.metrics_2cls import *
from collections import OrderedDict
from models.vit import vit_base_patch16_224

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOMATION_ROOT = os.path.dirname(SCRIPT_DIR)
TRAIN_DATA_ROOT = os.path.join(AUTOMATION_ROOT, 'data', 'mlkl', 'train', 'v1')

def get_cfg(fold):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='LSTM')
    parser.add_argument('--n_epochs', type=int, default=200)  
    parser.add_argument('--bt_size', type=int, default=32) 
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)') 
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(TRAIN_DATA_ROOT, '归一化插值后npz_1024_高斯平滑1'),
    )
    parser.add_argument(
        '--train_csv_dir',
        type=str,
        default=os.path.join(
            TRAIN_DATA_ROOT,
            '5折交叉验证_42',
            'train_fold{}_多标签.csv'.format(fold),
        ),
    )
    parser.add_argument(
        '--val_csv_dir',
        type=str,
        default=os.path.join(
            TRAIN_DATA_ROOT,
            '5折交叉验证_42',
            'val_fold{}_多标签.csv'.format(fold),
        ),
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        default=os.path.join(SCRIPT_DIR, 'model_hub'),
    )

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')

    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    parse_config = parser.parse_args()
    # Key: timm's create_scheduler requires args.epochs
    # syncs with the n_epochs we already have
    parse_config.epochs = parse_config.n_epochs
    return parse_config


    
def compute_metrics(outputs, targets, loss_fn):
    
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()

    targets1 = torch.argmax(targets, dim=1)
    loss = loss_fn(outputs, targets1).cpu().item()
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    #specificity = Specificity(outputs, targets)
    precision = Precision(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    specificity = spe(outputs, targets)
    metrics = OrderedDict([
        ('loss', loss),
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('kappa', kappa),
        ('confusion_matrix',cm),
        ('specificity',specificity)
    ])
        
    return metrics


# Define label smoothing function
def label_smoothing_loss(logits, target, smoothing):
    num_classes = logits.size(1)
    with torch.no_grad():
        # Use label smoothing
        smoothed_labels = torch.full(size=(target.size(0), num_classes), fill_value=smoothing / (num_classes - 1)).cuda()
        smoothed_labels.scatter_(dim=1, index=target.unsqueeze(1), value=1.0 - smoothing)

    log_prob = F.log_softmax(logits, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / logits.size(0)
    return loss

# -------------------------- train func --------------------------# 
def train(epoch, model, loss2_cls, loader_train):
    print(f"\n------ Training Epoch {epoch} ------")
    model.train()
    total_loss = 0

    for batch_idx, (data, _, label_cls_onehot, name) in enumerate(tqdm(loader_train)):
        img = data.cuda().float()
        label_cls_onehot = label_cls_onehot.cuda()

        pre = model(img)
        label_cls_onehot = torch.argmax(label_cls_onehot, dim=1)

        loss = loss2_cls(pre, label_cls_onehot)

        # cumulative loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(loader_train)
    print(f"Epoch {epoch} — Train Loss: {avg_loss:.4f}")
    return avg_loss
# -------------------------- test func --------------------------#
def test(epoch, model, loader_eval, loss_fn):
    print(f"\n------ Testing Epoch {epoch} ------")
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data, _, label_cls_onehot, name in tqdm(loader_eval):
            img = data.cuda().float()
            label_cls_onehot = label_cls_onehot.cuda()
            output = model(img)

            if isinstance(output, (tuple, list)):
                output = output[0]

            predictions.append(output)
            labels.append(label_cls_onehot)

    # Original indicator calculation
    evaluation_metrics = compute_metrics(predictions, labels, loss_fn)
    val_loss = evaluation_metrics['loss']
    val_acc = evaluation_metrics['acc']
    val_spe = evaluation_metrics['specificity']

    # New: print it out in Spyder
    print(f"Epoch {epoch} — Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

    # keeps the original 4 return values unchanged
    return evaluation_metrics, val_loss, val_acc, val_spe



if __name__ == '__main__':
    for fold in range(1,6):
        print("\n" + "="*30)
        print(f"Start training fold {fold}")
        print("="*30)
        # -------------------------- get args --------------------------#
        gc.collect()
        torch.cuda.empty_cache()
        parse_config = get_cfg(fold)
        # -------------------------- build dataloaders --------------------------#
        train_dataset = XrayDataset_train(parse_config)
        val_dataset = XrayDataset_val(parse_config)
        loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=parse_config.bt_size, shuffle=True, drop_last=True)
        loader_eval = torch.utils.data.DataLoader(val_dataset, batch_size=1)

        # -------------------------- build models --------------------------#
        model = vit_base_patch16_224(num_classes=2).cuda()
        loss2_cls = nn.CrossEntropyLoss()
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=parse_config))
        lr_scheduler, num_epochs = create_scheduler(parse_config, optimizer)
        lr_scheduler.step(0)

        # -------------------------- start training --------------------------#
        max_acc = 0
        learning_rates = []
        global_train_loss = []
        global_val_loss = []
        global_val_acc = []
        global_val_spe = []

        # -------------------------- build loggers and savers --------------------------#
        exp_name = f'fold{fold}'
        os.makedirs('./logs/{}/log'.format(exp_name), exist_ok=True)
        os.makedirs('./checkpoints/{}/'.format(exp_name), exist_ok=True)
        writer = SummaryWriter('./logs/{}/log'.format(exp_name))
        log_path = './logs/{}/log'.format(exp_name)
        out_images_path = './logs/{}/val_images'.format(exp_name)
        os.makedirs(out_images_path, exist_ok=True)
        save_path_acc = './checkpoints/{}/best_acc.pth'.format(exp_name)
        EPOCHS = parse_config.n_epochs
        logging.basicConfig(filename=os.path.join(log_path,'train_log.log'),
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        output_file = './logs/fold{}/参数.txt'.format(fold)  # Set output folder path
        # Open the file and write the parameters
        with open(output_file, 'w') as f:
            for arg_name in vars(parse_config):
                arg_value = getattr(parse_config, arg_name)
                f.write(f"{arg_name}: {arg_value}\n")

        # start training
        for epoch in range(1, EPOCHS + 1):
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            learning_rates.append(current_lr)
            start = time.time()
            train_loss = train(epoch, model, loss2_cls, loader_train)
            eval_metrics, val_loss, val_acc, val_spe = test(epoch, model, loader_eval, loss2_cls)
            global_train_loss.append(train_loss)
            global_val_loss.append(val_loss)
            global_val_acc.append(val_acc)
            global_val_spe.append(val_spe)

            if eval_metrics['acc'] > max_acc:   #At least the next model can be saved
                print("---save best acc model----")
                max_acc = eval_metrics['acc']
                max_acc = round(max_acc, 3)  
                torch.save(model.state_dict(), save_path_acc)

            num_updates = epoch * parse_config.bt_size
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics['acc'])

            time_elapsed = time.time() - start

            if epoch % 10 == 0:

                writer.add_scalar('fold{}/learning rate'.format(fold), current_lr, epoch)
                writer.add_scalar('fold{}/train_metrics/loss'.format(fold), train_loss, epoch)
                writer.add_scalar('fold{}/val_metrics/loss'.format(fold), eval_metrics['loss'], epoch)
                writer.add_scalar('fold{}/val_metrics/acc'.format(fold), eval_metrics['acc'], epoch)
                writer.add_scalar('fold{}/val_metrics/f1'.format(fold), eval_metrics['f1'], epoch)
                logging.info(
                    '\n valid/acc:{} \n valid/specificity:{} \n confusion_matrix:{} \n'.
                        format(
                            eval_metrics['acc'], 
                            eval_metrics['specificity'], 
                            eval_metrics['confusion_matrix']))
        

        # Creates a list of tuples containing indicator name, value and index
        metrics_values_and_indices = [
            ('val_acc', global_val_acc),
            ('val_spe', global_val_spe),
        ]
        # Traverse each indicator, obtain the top three values and indexes, and output the results
        for metric, values in metrics_values_and_indices:
            max_values = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:3]
            
            # output result
            for i, (value, index) in enumerate(max_values):
                logging.info("Top {} - {}: {} (Index: {})".format(i + 1, metric, value, index))


        # -------------------------- Draw learning rate curve --------------------------#
        plt.clf()
        plt.plot(learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        output_directory = './logs/fold{}'.format(fold)  # Set output folder path
        os.makedirs(output_directory, exist_ok=True)
        image_filename = os.path.join(output_directory, 'learning_rate_schedule.png')
        plt.savefig(image_filename)


        # -------------------------- Draw train_loss and val_loss curves --------------------------#
        plt.clf()
        # Make sure the x-axis value matches the number of data points
        plt.figure(figsize=(10, 8))
        epochs = max(len(global_train_loss), len(global_val_loss))
        x = range(epochs)
        # Draw the training loss curve, using a thick red solid line
        plt.plot(x[:len(global_train_loss)], global_train_loss, label='Train Loss', color='red', linewidth=2)

        # Draw the verification loss curve, using a thick blue solid line
        plt.plot(x[:len(global_val_loss)], global_val_loss, label='Validation Loss', color='blue', linewidth=2)

        # Add legend, title and axis labels
        # plt.legend()
        # plt.title('Training and Validation Loss')
        # plt.xlabel('Epochs or Steps')
        # plt.ylabel('Loss')
        # Save chart
        image_filename = os.path.join(output_directory, 'train_val_loss.png')
        plt.savefig(image_filename)
        # -------------------------- Save the loss value to a text file --------------------------#
        train_loss_filename = os.path.join(output_directory, 'global_train_loss.txt')
        val_loss_filename = os.path.join(output_directory, 'global_val_loss.txt')

        with open(train_loss_filename, 'w') as f:
            for loss in global_train_loss:
                f.write(f"{loss}\n")

        with open(val_loss_filename, 'w') as f:
            for loss in global_val_loss:
                f.write(f"{loss}\n")

        # -------------------------- Draw train_loss curve --------------------------#
        plt.clf()  
        x = range(len(global_train_loss))
        plt.plot(x, global_train_loss, label='Train Loss', linestyle='--')
        plt.legend()
        plt.title('Training Loss')
        plt.xlabel('Epochs or Steps')
        plt.ylabel('Loss')
        image_filename = os.path.join(output_directory, 'trainLoss.png')
        plt.savefig(image_filename)

        # -------------------------- Draw val_loss curve --------------------------#
        plt.clf()  
        x = range(len(global_val_loss))
        plt.plot(x, global_val_loss, label='Validation Loss', linestyle='--')
        plt.legend()
        plt.title('Validation Loss')
        plt.xlabel('Epochs or Steps')
        plt.ylabel('Loss')
        image_filename = os.path.join(output_directory, 'valLoss.png')
        plt.savefig(image_filename)


        # -------------------------- Draw acc curve --------------------------#
        plt.clf()  
        x = range(len(global_val_acc))
        plt.plot(x, global_val_acc, label='Validation Acc', linestyle='--')
        plt.legend()
        plt.title('Training and Validation Acc')
        plt.xlabel('Epochs or Steps')
        plt.ylabel('Acc')
        image_filename = os.path.join(output_directory, 'acc.png')
        plt.savefig(image_filename)



    
