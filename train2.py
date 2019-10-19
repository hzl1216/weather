from __future__ import division
from __future__ import print_function

import time
import random
import argparse
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from util import  accuracy, k_fold
from torch.optim.lr_scheduler import *
from net import ResNet50,ResNet101,ResNet152,Net,EfficientNet_new
from dataset import ProductDataset,TestDataset
import torchvision.models as models
from optimization import *
import pretrainedmodels
from train_tool import pencil_train
from efficientnet_pytorch import EfficientNet
from losses import LabelSmoothSoftmaxCE
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)



transforms_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

kwargs = {'num_workers': 4, 'pin_memory': True}



model_settings = {
    'se_resnext50_32x4d':{
        'epochs': 50,
        'lr': 0.0001,
        'dropout': 0.2,
        'use_pencil_train': True,
        'stage1': 10,
        'stage2': 35,
        'use_warm_up': True,
        'batch_size': 100,
        'resize_shape': 256,
        'input_size': 224,
        'transforms_norm': transforms_norm
    },
    'se_resnet101': {
        'epochs': 50,
        'lr': 0.0001,
        'dropout': 0.5,
        'use_pencil_train': True,
        'stage1': 10,
        'stage2': 35,
        'use_warm_up': True,
        'batch_size': 100,
        'resize_shape': 256,
        'input_size': 224,
        'transforms_norm': transforms_norm
    },
    'se_resnext101_32x4d': {
        'epochs': 50,
        'lr': 2e-4,
        'dropout': 0.2,
        'use_pencil_train': True,
        'stage1': 10,
        'stage2': 35,
        'use_warm_up': True,
        'batch_size': 100,
        'resize_shape': 256,
        'input_size': 224,
        'transforms_norm': transforms_norm
    },
    'se_resnet50': {
        'epochs': 50,
        'lr': 0.0001,
        'dropout': 0.2,
        'use_pencil_train': True,
        'stage1': 10,
        'stage2': 35,
        'use_warm_up': True,
        'batch_size': 100,
        'resize_shape': 256,
        'input_size': 224,
        'transforms_norm': transforms_norm
    },
    'efficientnet-b5':{
        'epochs': 50,
        'lr': 2e-4,
        'dropout': 0.2,
        'use_pencil_train': True,
        'stage1': 10,
        'stage2': 35,
        'use_warm_up': True,
        'batch_size': 50,
        'resize_shape': 256,
        'input_size': 224,
        'transforms_norm': transforms_norm
    },
    'efficientnet-b0':{
        'epochs': 50,
        'lr': 0.001,
        'dropout': 0.2,
        'use_pencil_train': True,
        'stage1': 10,
        'stage2': 35,
        'use_warm_up': True,
        'batch_size': 200,
        'resize_shape': 256,
        'input_size': 224,
        'transforms_norm': transforms_norm
    },
    'efficientnet-b4':{
        'epochs': 50,
        'lr': 2e-4,
        'dropout': 0.2,
        'use_pencil_train': True,
        'stage1': 10,
        'stage2': 35,
        'use_warm_up': True,
        'batch_size': 50,
        'resize_shape': 256,
        'input_size': 224,
        'transforms_norm': transforms_norm
     }
}
def generator_label():
    model_name='se_resnet101'
    model_setting = model_settings[model_name]


    net = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = Net(net, drop=model_setting['dropout'])


    optimizer = AdamW(model.parameters(), lr=model_setting['lr'], weight_decay=5e-4)
    warmup_step = model_setting['epochs'] * 300 // model_setting['batch_size']
    totals = model_setting['epochs'] * 5000 // model_setting['batch_size']
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_step, t_total=totals)

    loss = nn.CrossEntropyLoss().cuda()
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model)


    # Load data
    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(model_setting['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        model_setting['transforms_norm']
    ])

    train_dataset = ProductDataset('Train_label.csv', 'train',
                                   # train_dataset = ProductDataset('train_label_modify.csv','train',
                                   transform=train_transforms)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=model_setting['batch_size'],
                                               shuffle=True,
                                               **kwargs
                                               )

    t_total = time.time()
    y = []
    for epoch in range(model_setting['epochs']):
        #            train(epoch,train_loader)

        y = pencil_train(train_loader, model, loss, optimizer, scheduler, epoch, y,
                             model_setting['stage1'], model_setting['stage2'], len(train_dataset), 9)

    print('Generator Finished!')
    y_file = "y.npy"
    y = np.load(y_file)
    y = np.argmax(y, 1)
    y = [x + 1 for x in y]

    label_file = 'Train_label.csv'
    label_file = pd.read_csv(label_file)
    file_names = label_file["FileName"]
    dataframe = pd.DataFrame({'FileName': file_names, 'type': y})
    dataframe.to_csv('new_Train_label.csv', index=False, sep=',')
def train(epoch, data_loader):
    t = time.time()
    model.train()
    avg_loss = []
    #    scheduler.step()
    #    print("Decaying learning rate to %g" % scheduler.get_lr()[0])
    for i, data in enumerate(data_loader):
        images, labels, _ = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        output = model(images)

        loss_train = loss(output, labels)
        loss_train.backward()
        avg_loss.append(loss_train.item())
        optimizer.step()
        scheduler.step()
    print('\nEpoch train: %d,loss: %f' % (epoch, sum(avg_loss) / len(avg_loss)))


def val(epoch, data_loader):
    print("\nValidation Epoch: %d" % epoch)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            images, labels, _ = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            out = model(images)

            _, predicted = torch.max(out.data, 1)

            total += images.size(0)
            correct += predicted.data.eq(labels.data).cpu().sum()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))
    return 1.0 * correct.numpy() / total

def test(epoch, test_loader, file_name):
    print("\nstart test")
    model.eval()

    types = []
    softmax = nn.Softmax(dim=1).cuda()

    with torch.no_grad():
        all_output = None
        for _ in range(8):
            file_names = []
            outputs = None
            for batch_idx, data in enumerate(test_loader):
                images, images_names = data
                if torch.cuda.is_available():
                    images = images.cuda()
                out = model(images)
                out = softmax(out).cpu().numpy()
                if outputs is not None:
                    outputs = np.concatenate((outputs, out))
                else:
                    outputs = out
                file_names += images_names
            if all_output is not None:
                all_output += outputs
            else:
                all_output = outputs
        predicted = np.argmax(all_output, 1)
        types += [i + 1 for i in predicted.tolist()]

    dataframe = pd.DataFrame({'FileName': file_names, 'type': types})
    dataframe.to_csv(file_name, index=False, sep=',')




if __name__ == '__main__':
    generator_label()
    model_names= ['se_resnet101']
#    model_names = ['efficientnet-b5','se_resnext101_32x4d']
#    model_names = ['efficientnet-b4','se_resnet50']
    for model_name in model_names:
        k_folds = k_fold(6391, 5)
        model_setting=model_settings[model_name]
        avg_best_acc = []
        for i in range(5):
            if 'efficientnet' in model_name:
                model = EfficientNet_new.from_pretrained(model_name=model_name,num_classes=9,dropout_rate=model_setting['dropout'])
            else:
                net = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
                #net = pretrainedmodels.__dict__[model_name](num_classes=1000)
            # model=net
                model = Net(net,  drop=model_setting['dropout'])

            # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # 设置训练细节
            optimizer = AdamW(model.parameters(), lr= model_setting['lr'], weight_decay=5e-4)
            warmup_step = model_setting['epochs']*300//model_setting['batch_size']
            totals = model_setting['epochs']*5000//model_setting['batch_size']
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_step, t_total=totals)
#            loss = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3).cuda()
            loss = nn.CrossEntropyLoss().cuda()
            if torch.cuda.is_available():
                model.cuda()
                model = nn.DataParallel(model)
            if 0:
                model.load_state_dict(torch.load('model/%s_100.model'%model_name))
            train_index = k_folds[i][0]
            test_index = k_folds[i][1]
        # Load data
            train_transforms = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(model_setting['input_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                model_setting['transforms_norm']
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(model_setting['resize_shape']),
                transforms.CenterCrop(model_setting['input_size']),
                transforms.ToTensor(),
                model_setting['transforms_norm']
            ])
            train_dataset = ProductDataset('new_Train_label.csv', 'train',train_index,
                                           # train_dataset = ProductDataset('train_label_modify.csv','train',
                                           transform=train_transforms)
            val_dataset = ProductDataset('new_Train_label.csv', 'train',test_index,
                                         transform=test_transforms)


            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                      batch_size=model_setting['batch_size'],
                                                      shuffle=True,
                                                      **kwargs
                                                      )
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=model_setting['batch_size'],
                                                     shuffle=False,
                                                     **kwargs
                                                    )
            test_dataset = TestDataset('test',transform=train_transforms)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=model_setting['batch_size'],
                                                      shuffle=False,
                                                      **kwargs
                                                      )
            # Model and optimizer
            # Train model
            t_total = time.time()
            y = []
            best_acc = 0
            last_best_epoch = model_setting['stage1']
            for epoch in range(model_setting['epochs']):
                train(epoch,train_loader)
                acc = val(epoch,val_loader)
                if epoch>model_setting['stage1'] and acc > best_acc :
                    best_acc = acc
                    last_best_epoch=epoch
                    torch.save(model.state_dict(), 'model/%s_%d_best.model' % (model_name,i))
                if epoch==model_setting['epochs']-1 or epoch-last_best_epoch>20:
                    test_file_name = 'result/submit_%s_%d_final.csv' % (model_name, i)
                    test(1 + epoch, test_loader, test_file_name)

                    torch.save(model.state_dict(), 'model/%s_%d_final.model' % (model_name,i))
                    avg_best_acc.append(best_acc)
                    model.load_state_dict(torch.load('model/%s_%d_best.model' % (model_name,i)))
                    test_file_name = 'result/submit_%s_%d_best.csv'%(model_name,i)
                    test(1 + epoch,test_loader,test_file_name)
                    break
            print('Optimization Finished!')
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print('-----------%s the average best accuracy is %f'% (model_name, sum(avg_best_acc)/(len(avg_best_acc))))
        print('--------------------------------------------------')
