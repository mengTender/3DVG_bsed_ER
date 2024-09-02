import numpy as np
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from tensorboardX import SummaryWriter
import scipy.io
from sklearn.metrics import average_precision_score, precision_recall_curve

from datasets import Emotic_CSVDataset
from loss_funcs import MultiplicativeLoss, DiscreteLoss
from models.model import ModelForER, collate_fn


def train_data(opt, scheduler, model, device, train_loader, val_loader, multip_loss, disc_loss, train_writer, val_writer, model_path, args):

    model.to(device)
    print('starting training')
    for e in range(args.epochs):

        running_loss = 0.0
        running_cat_loss = 0.0
        running_multip_loss = 0.0

        model.train()

        # train models for one epoch
        for data, labels_cat, labels_cont in iter(train_loader):
            for key in data.keys():
                data[key] = data[key].to(device)
            labels_cat = labels_cat.to(device)

            opt.zero_grad()
            peb_out, ceb_out, pred_out = model(data)

            cat_loss_batch = disc_loss(pred_out, labels_cat)

            loss = (args.cat_loss_weight * cat_loss_batch) + (args.multip_loss_weight * multip_loss)

            running_loss += loss.item()
            running_cat_loss += cat_loss_batch.item()
            running_multip_loss += multip_loss.item()

            loss.backward()
            opt.step()

        if e % 1 == 0:
            print('epoch = %d loss = %.4f cat loss = %.4f cont_loss = %.4f' % (
            e, running_loss, running_cat_loss, running_cont_loss))

        train_writer.add_scalar('losses/total_loss', running_loss, e)
        train_writer.add_scalar('losses/categorical_loss', running_cat_loss, e)
        train_writer.add_scalar('losses/continuous_loss', running_cont_loss, e)

        running_loss = 0.0
        running_cat_loss = 0.0
        running_cont_loss = 0.0

        model.eval()

        with torch.no_grad():
            # validation for one epoch
            for data, labels_cat, _ in iter(val_loader):
                for key in data.keys():
                    data[key] = data[key].to(device)
                labels_cat = labels_cat.to(device)

                peb_out, ceb_out, pred_out = model(data)

                cat_loss_batch = disc_loss(pred_out, labels_cat)

                loss = (args.cat_loss_weight * cat_loss_batch) + (args.multip_loss_weight * multip_loss)

                running_loss += loss.item()
                running_cat_loss += cat_loss_batch.item()
                running_multip_loss += multip_loss.item()

        if e % 1 == 0:
            print('epoch = %d validation loss = %.4f cat loss = %.4f cont loss = %.4f ' % (
            e, running_loss, running_cat_loss, running_cont_loss))

        val_writer.add_scalar('losses/total_loss', running_loss, e)
        val_writer.add_scalar('losses/categorical_loss', running_cat_loss, e)
        val_writer.add_scalar('losses/continuous_loss', running_cont_loss, e)

        scheduler.step()

    print('completed training')
    model.to("cpu")
    torch.save(model, os.path.join(model_path, 'model_emotic.pth'))
    print('saved models')


# Categorical emotion classes
cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']


def train_emotic(result_path, model_path, train_log_path, val_log_path, args):
    ''' Prepare dataset, dataloders, models.
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training.
    :param train_log_path: Directory path to save the training logs.
    :param val_log_path: Directoty path to save the validation logs.
    :param args: Runtime arguments.
    '''
    # Load preprocessed data from npy files
    data_root = './datasets/emoticon'
    train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_root, 'test.csv'))
    val_df = pd.read_csv(os.path.join(data_root, 'val.csv'))

    # Initialize Dataset and DataLoader
    train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                          transforms.ToTensor()])
    # test_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])

    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    train_dataset = Emotic_CSVDataset(train_df, cat2ind, train_transform, data_root, train='train')
    val_dataset = Emotic_CSVDataset(val_df, cat2ind, train_transform, data_root, train='val')
    # train_dataset = Emotic_CSVDataset(train_df, cat2ind, train_transform, data_root, train='train')

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)

    print('train loader ', len(train_loader), 'val loader ', len(val_loader))

    # Prepare models

    model_er = ModelForER(out_feature=26)

    for param in model_er.parameters():
        param.requires_grad = True

    device = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")
    opt = optim.Adam(model_er, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(opt, step_size=7, gamma=0.1)
    disc_loss = DiscreteLoss(args.discrete_loss_weight_type, device)
    multip_loss = MultiplicativeLoss(2)

    train_writer = SummaryWriter(train_log_path)
    val_writer = SummaryWriter(val_log_path)

    # training
    train_data(opt, scheduler, model_er, device, train_loader, val_loader, multip_loss, disc_loss,
               train_writer, val_writer, model_path, args)
    # validation
    test_data(model_er, device, val_loader, ind2cat, len(val_dataset),
              result_dir=result_path, test_type='val')


def test_scikit_ap(cat_preds, cat_labels, ind2cat):
    '''
    Calculate average precision per emotion category using sklearn library.
    '''
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
        print('Category %16s %.5f' % (ind2cat[i], ap[i]))
    print('Mean AP %.5f' % (ap.mean()))
    return ap


def get_thresholds(cat_preds, cat_labels):

    thresholds = np.zeros(26, dtype=np.float32)
    for i in range(26):
        p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
        for k in range(len(p)):
            if p[k] == r[k]:
                thresholds[i] = t[k]
                break
    return thresholds


def test_data(model, device, data_loader, ind2cat, num_images, result_dir='./', test_type='val'):
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))

    with torch.no_grad():
        model.to(device)
        indx = 0
        print('starting testing')
        for data, labels_cat, _  in iter(data_loader):
            for key in data.keys():
                data[key] = data[key].to(device)
            labels_cat = labels_cat.to(device)

            peb_out, ceb_out, pred_out = model(data)

            cat_preds[indx: (indx + pred_out.shape[0]), :] = pred_out.to("cpu").data.numpy()
            cat_labels[indx: (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            indx = indx + pred_out.shape[0]

    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    print('completed testing')

    # Mat files used for emotic testing (matlab script)
    scipy.io.savemat(os.path.join(result_dir, '%s_cat_preds.mat' % (test_type)), mdict={'cat_preds': cat_preds})
    scipy.io.savemat(os.path.join(result_dir, '%s_cat_labels.mat' % (test_type)), mdict={'cat_labels': cat_labels})
    print('saved mat files')

    test_scikit_ap(cat_preds, cat_labels, ind2cat)
    thresholds = get_thresholds(cat_preds, cat_labels)
    np.save(os.path.join(result_dir, '%s_thresholds.npy' % (test_type)), thresholds)
    print('saved thresholds')


def test_emotic(result_path, model_path, args, data_root='./datasets/emotic'):
    # Prepare models
    model = torch.load(os.path.join(model_path, 'model_emotic.pth'))
    print('Succesfully loaded models')

    # Load data preprocessed npy files
    test_df = pd.read_csv(os.path.join(data_root, 'test.csv'))

    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    # Initialize Dataset and DataLoader
    test_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = Emotic_CSVDataset(test_df, cat2ind, test_transform, data_root, train='test')
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    print('test loader ', len(test_loader))

    device = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")
    test_data(model, device, test_loader, ind2cat, len(test_dataset),
              result_dir=result_path, test_type='test')


