import pandas as pd
from model_settings import model_settings
from util import k_fold
import os
import numpy as np
from net import ResNet50,ResNet101,ResNet152,Net,EfficientNet_new,pnasnet
from dataset import ProductDataset,TestDataset
from optimization import *
import pretrainedmodels
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import time
from train_tool import pencil_train,train,test,val
kwargs = {'num_workers': 12, 'pin_memory': True}


def select_samples(label_file_name='Train_label.csv',select_labels=[6,8]):
    if label_file_name=='Train_label.csv':
        generator_file_name='train_next.csv'
    else:
        generator_file_name = 'test_next.csv'
    label_file = pd.read_csv(label_file_name)
    new_labels = []
    new_filenames =[]
    labels=label_file['type']
    file_names=label_file["FileName"]
    for i in  range(len(labels)):
        if labels[i] in  select_labels:
            new_filenames.append(file_names[i])
            new_labels.append(labels[i])
    dataframe = pd.DataFrame({'FileName': new_filenames, 'type': new_labels})
    dataframe.to_csv(generator_file_name, index=False, sep=',')
select_samples()
select_samples('result/se_resnext101_32x4d_submit.csv')

if __name__ == '__main__':
    train_label_file_name= 'train_next.csv'
    test_label_file_name = 'test_next.csv'
    label_file = pd.read_csv(train_label_file_name)
    model_name = 'train_next'
    model_setting = model_settings[model_name]
    k_file = 'train_next_k_folds.npy'
    if os.path.exists(k_file):
        k_folds = np.load(k_file, allow_pickle=True)
    else:
        k_folds = k_fold(len('FileName'), 5)
        np.save(k_file, k_folds)
    avg_best_acc = []
    for i in range( 5):
        # net = pretrainedmodels.__dict__[model_name](num_classes=1000)
        # model = Net(net, final_feature=model_setting['final_feature'], dropout=model_setting['dropout'])
        model = ResNet50(2)
        optimizer = AdamW(model.parameters(), lr=model_setting['lr'], weight_decay=1e-4)
        warmup_step = model_setting['epochs'] * 300 // model_setting['batch_size']
        totals = model_setting['epochs'] * 5000 // model_setting['batch_size']
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_step, t_total=totals)
        loss = nn.CrossEntropyLoss().cuda()
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
        train_index = k_folds[i][0]
        test_index = k_folds[i][1]
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
        train_dataset = ProductDataset(train_label_file_name, 'train', train_index,
                                       transform=train_transforms)
        val_dataset = ProductDataset(train_label_file_name, 'train', test_index,
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
        test_dataset = ProductDataset(test_label_file_name,'test', transform=train_transforms)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=model_setting['batch_size'],
                                                  shuffle=False,
                                                  **kwargs
                                                  )
        t_total = time.time()
        y = []
        best_acc = 0
        last_best_epoch = model_setting['stage1']
        for epoch in tqdm(range(model_setting['epochs'])):
            if model_setting['use_pencil_train']:
                pass
                y = pencil_train(train_loader, model, loss, optimizer, scheduler, epoch, y, model_setting['stage1'],
                                 model_setting['stage2'], len(train_dataset), 2)
            else:
                train(epoch, train_loader, loss, optimizer, scheduler)
            acc = val(epoch, model, val_loader)
            if epoch > 20 and acc > best_acc:
                best_acc = acc
                last_best_epoch = epoch
                torch.save(model.state_dict(), 'model/train_next_%d.model' % ( i))
            if epoch == model_setting['epochs'] - 1 or epoch - last_best_epoch > 20:
                avg_best_acc.append(best_acc)
                model.load_state_dict(torch.load('model/train_next_%d.model' % (i)))
                test_file_name = 'result/submit_next_%d.csv' % ( i)
                test_file_name_list = test_dataset.file_name
                test(model, test_loader, test_file_name_list, test_file_name, 8)
                break
        print('Optimization Finished!')
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('-----------%s the average best accuracy is %f' % (model_name, sum(avg_best_acc) / (len(avg_best_acc))))
    print('--------------------------------------------------')