from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import time
import random
import argparse
import numpy as np
from torchvision import transforms
import torch.nn as nn
from util import  accuracy, k_fold
from net import ResNet50,ResNet101,ResNet152,Net,EfficientNet_new,pnasnet
from dataset import ProductDataset,TestDataset
from optimization import *
import pretrainedmodels
from train_tool import pencil_train,train,test,val,generator_label
import os
from model_settings import model_settings

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

#random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)



transforms_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

transforms_norm2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
kwargs = {'num_workers': 12, 'pin_memory': True}




if __name__ == '__main__':
#    model_names= ['senet154','se_resnet101','efficientnet-b5','efficientnet-b4','se_resnext101_32x4d','se_resnext50_32x4d','se_resnet50']
    model_names= ['resnext101_32x8d_wsl']
    for model_index,model_name in enumerate (model_names):
        train_label = 'Train_label.csv'
        model_setting=model_settings[model_name]
        k_file = '%s_k_folds.npy'%model_name
        if os.path.exists(k_file):
            k_folds=np.load(k_file,allow_pickle=True)
        else:
            k_folds = k_fold(6391, 5)
            np.save(k_file, k_folds)

        avg_best_acc = []
        for i in range(3,5):
            print('-------------------------train %d_fold'%i)
            if 'efficientnet' in model_name:
                model = EfficientNet_new.from_pretrained(model_name=model_name,num_classes=9,dropout_rate=model_setting['dropout'])
            elif 'resnext101_32x8d_wsl' in model_name:
                net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
                model = Net(net,final_feature=model_setting['final_feature'],dropout=model_setting['dropout'])
            else:
                net = pretrainedmodels.__dict__[model_name](num_classes=1000)
                model = Net(net,final_feature=model_setting['final_feature'],dropout=model_setting['dropout'])
            optimizer = AdamW(model.parameters(), lr= model_setting['lr'], weight_decay=1e-4)
            warmup_step = model_setting['epochs']*300//model_setting['batch_size']
            totals = model_setting['epochs']*5000//model_setting['batch_size']
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_step, t_total=totals)
            loss = nn.CrossEntropyLoss().cuda()
            if torch.cuda.is_available():
                model.cuda()
                model = nn.DataParallel(model)
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
            train_dataset = ProductDataset(train_label, 'train',train_index,
                                           transform=train_transforms)
            val_dataset = ProductDataset(train_label, 'train',test_index,
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
            for epoch in tqdm(range(model_setting['epochs'])):
                if model_setting['use_pencil_train']:
                    pass
                    y=pencil_train(train_loader,model,loss,optimizer,scheduler,epoch,y,model_setting['stage1'],model_setting['stage2'],len(train_dataset),9)
                else:
                    train(epoch, train_loader,loss,optimizer,scheduler)
                acc = val(epoch, model,val_loader)
                if epoch>20 and acc > best_acc :
                    best_acc = acc
                    last_best_epoch=epoch
                    torch.save(model.state_dict(), 'model/%s_%d_best.model' % (model_name,i))
                if epoch==model_setting['epochs']-1 or epoch - last_best_epoch >20:
#                    test_file_name = 'result/submit_%s_%d_final.csv' % (model_name, i)
#                    test(model, test_loader, test_file_name,8)

                    torch.save(model.state_dict(), 'model/%s_%d_final.model' % (model_name,i))
                    avg_best_acc.append(best_acc)
                    model.load_state_dict(torch.load('model/%s_%d_best.model' % (model_name,i)))
                    test_file_name = 'result/submit_%s_%d_best.csv'%(model_name,i)
                    test_file_name_list = test_dataset.file_name
                    test(model,test_loader,test_file_name_list,test_file_name,8)
                    break
            print('Optimization Finished!')
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print('-----------%s the average best accuracy is %f'% (model_name, sum(avg_best_acc)/(len(avg_best_acc))))
        print('--------------------------------------------------')
        generator_label(model_name, k_folds,model_settings,kwargs)
                
