import argparse
import logging
import os
import random

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

from datasets import RafDataSet
from losses import BalanceLoss, CenterLoss, CompactnessLoss
from networks.fdrl import FDRL


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# setup_seed(456)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_1', help='Name of experiment.')
    parser.add_argument('--raf_path', type=str, default='../DAN/datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate for Adam.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_class', type=int, default=7, help='Number of class.')
    parser.add_argument('--num_branch', type=int, default=9, help='Number of branches.')
    parser.add_argument('--feat_dim', type=int, default=128, help='Number of branches.')
    parser.add_argument('--lambda_1', type=float, default=1e-4, help='Coefficient for compactness loss.')
    parser.add_argument('--lambda_2', type=float, default=1, help='Coefficient for balance loss.')
    parser.add_argument('--lambda_3', type=float, default=1e-4,help='Coefficient for distribution loss.')
    parser.add_argument('--resume', type=str, default=None, help='Path of weight to be resumed.')
    return parser.parse_args()


def run_training():
    args = parse_args()
    log_name = args.exp_name
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', 
    datefmt='%Y-%m-%d %H:%M', handlers=[logging.FileHandler('./logs/' + log_name + '.log', 'w', 'utf-8')])
    logging.getLogger('numexpr').setLevel(logging.WARNING)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = FDRL(args.num_branch, 512, args.feat_dim, args.num_class)
    if args.resume != None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

    model.to(device)

    # Additional augmentation used
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(224, padding=32)],
            p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms)
    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_dataset = RafDataSet(args.raf_path, phase='test',transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
    num_workers=args.workers, shuffle=False, pin_memory=True)

    # Define losses
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_c = CompactnessLoss(args.num_branch, args.feat_dim)
    criterion_d = CenterLoss(args.num_class, args.num_branch)
    criterion_b = BalanceLoss(args.num_branch)


    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    if args.resume != None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 18, 25, 32], gamma=0.1)

    # training loop
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()


        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device).long()

            fdn_feat, alphas, pred = model(imgs)
            loss = 1 * criterion_cls(pred, targets) \
                + args.lambda_1 * criterion_c(fdn_feat) \
                + args.lambda_2 * criterion_b(alphas) \
                + args.lambda_3 * criterion_d(alphas, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicts = torch.max(pred, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
            epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
        logging.info('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
            epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
        
        # evaluate at every epoch
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device).long()

                fdn_feat, alphas, pred = model(imgs)
                loss = 1 * criterion_cls(pred, targets) \
                    + args.lambda_1 * criterion_c(fdn_feat) \
                    + args.lambda_2 * criterion_b(alphas) \
                    + args.lambda_3 * criterion_d(alphas, targets)

                running_loss += loss.item()
                iter_cnt += 1
                _, predicts = torch.max(pred, 1)
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(predicts.cpu().numpy().tolist())
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += pred.size(0)

            running_loss = running_loss/iter_cnt
            scheduler.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            bacc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            best_acc = max(acc, best_acc)
            logging.info("[Epoch %d] Validation accuracy:%.4f. Balanced Accuracy:%.4f. Loss:%.3f" % (
                epoch, acc, bacc, running_loss))
            logging.info("Best_acc:" + str(best_acc))
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Balanced Accuracy:%.4f. Loss:%.3f" % (
                epoch, acc, bacc, running_loss))
            tqdm.write("Best_bacc:" + str(best_acc))

            if bacc == best_acc:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('checkpoints', "" + log_name + "_best.pth"))
                tqdm.write('Best Model saved.')
            
            torch.save({'iter': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), },
                       os.path.join('checkpoints', "" + log_name + "_latest.pth"))
            tqdm.write('Latest Model saved.')


if __name__ == "__main__":
    run_training()
