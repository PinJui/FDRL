import argparse

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from datasets import RafDataSet
from losses import BalanceLoss, CenterLoss, CompactnessLoss
from networks.fdrl import FDRL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='../DAN/datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_class', type=int, default=7, help='Number of class.')
    parser.add_argument('--num_branch', type=int, default=9, help='Number of branches.')
    parser.add_argument('--feat_dim', type=int, default=128, help='Number of branches.')
    parser.add_argument('--lambda_1', type=float, default=1e-4, help='Coefficient for compactness loss.')
    parser.add_argument('--lambda_2', type=float, default=1, help='Coefficient for balance loss.')
    parser.add_argument('--lambda_3', type=float, default=1e-4, help='Coefficient for distribution loss.')
    parser.add_argument('--resume', type=str, default=None, help='Path of pretrained weight')
    return parser.parse_args()

args = parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FDRL(args.num_branch, 512, args.feat_dim, args.num_class)
ckpt = torch.load(args.resume, map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
model.to(device)

data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

val_dataset = RafDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)   
print('Validation set size:', val_dataset.__len__())   
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.workers, shuffle = False, pin_memory = True)

criterion_cls = torch.nn.CrossEntropyLoss()
criterion_c = CompactnessLoss(args.num_branch, args.feat_dim)
criterion_d = CenterLoss(args.num_class, args.num_branch)
criterion_b = BalanceLoss(args.num_branch)

running_loss = 0.0
iter_cnt = 0
bingo_cnt = 0
sample_cnt = 0
y_true = []
y_pred = []

model.eval()
for (imgs, targets) in val_loader:
    imgs = imgs.to(device)
    targets = targets.to(device)

    fdn_feat, alphas, pred = model(imgs)

    loss = 1 * criterion_cls(pred, targets) \
        + args.lambda_1 * criterion_c(fdn_feat) \
        + args.lambda_2 * criterion_b(alphas) \
        + args.lambda_3 * criterion_d(alphas, targets)

    running_loss += loss.item()
    iter_cnt+=1
    _, predicts = torch.max(pred, 1)
    correct_num  = torch.eq(predicts,targets)
    y_true.extend(list(targets.cpu().numpy()))
    y_pred.extend(list(predicts.cpu().numpy()))
    bingo_cnt += correct_num.sum().cpu()
    sample_cnt += pred.size(0)

running_loss = running_loss/iter_cnt
acc = bingo_cnt.float()/float(sample_cnt)
acc = np.around(acc.numpy(),4)
conf_mat = confusion_matrix(y_true, y_pred)
print('Testing loss:', running_loss)
print('Confusion matrix:')
print(conf_mat)
print('Testing acc:', acc)
print('Finish testing!')
