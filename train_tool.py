import time
import numpy as np
from torchvision import transforms
import torch.nn as nn
import pandas as pd
from net import ResNet50,ResNet101,ResNet152,Net,EfficientNet_new,pnasnet
from dataset import ProductDataset,TestDataset
from optimization import *
import pretrainedmodels

import os

def loss_function(criterion,output,target,target_var,index,new_y,y,epoch,stage1,stage2,alpha=0.4,beta=0.1):
    logsoftmax = nn.LogSoftmax(dim=1).cuda()
    softmax = nn.Softmax(dim=1).cuda()
    yy=None
    if epoch < stage1:
        # lc is classification loss
        lc = criterion(output, target_var)
        # init y_tilde, let softmax(y_tilde) is noisy labels
        onehot = torch.zeros(target.size(0), 9).scatter_(1, target.view(-1, 1), 10.0)
        onehot = onehot.numpy()
        new_y[index, :] = onehot
    else:
        yy = y
        yy = yy[index, :]
        yy = torch.FloatTensor(yy)
        yy = yy.cuda(non_blocking=True)
        yy = torch.autograd.Variable(yy, requires_grad=True)
        # obtain label distributions (y_hat)
        last_y_var = softmax(yy)
        lc = torch.mean(softmax(output) * (logsoftmax(output) - torch.log((last_y_var))))
        # lo is compatibility loss
        lo = criterion(last_y_var, target_var)
    # le is entropy loss
    le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))
    
    if epoch < stage1:
        loss = lc
    elif epoch < stage2:
        loss = lc + alpha * lo + beta * le
    else:
        loss = lc

    return loss,new_y,yy

def pencil_train(train_loader, model, criterion, optimizer,scheduler, epoch, y,stage1,stage2,datanum,classnum,alpha=0.4,beta=0.1,lambda1=600):


    # switch to train mode
    model.train()

    # new y is y_tilde after updating
    new_y = np.zeros([datanum,classnum])
    avg_loss = []
    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time

        index = index.numpy()

        target1 = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target1)

        # compute output
        output = model(input_var)
        loss,new_y,yy= loss_function(criterion,output,target,target_var,index,new_y,y,epoch,stage1,stage2)

        avg_loss.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if epoch >= stage1 and epoch < stage2:

            # update y_tilde by back-propagation
            yy.data.sub_(lambda1*yy.grad.data)

            new_y[index,:] = yy.data.cpu().numpy()


    print('\nEpoch train: %d,loss: %f' % (epoch,sum(avg_loss)/len(avg_loss)))

    if epoch < stage2:
        # save y_tilde
        y = new_y
        y_file = "y.npy"
        np.save(y_file,y)
        y_record =  "record/y_%03d.npy" % epoch
        np.save(y_record,y)
    return y

def train(epoch, model,data_loader,criterion,optimizer,scheduler):
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

        loss_train = criterion(output, labels)
        loss_train.backward()
        avg_loss.append(loss_train.item())
        optimizer.step()
        scheduler.step()
    print('\nEpoch train: %d,loss: %f' % (epoch, sum(avg_loss) / len(avg_loss)))

def generator_label(model_name,k_folds,model_settings,kwargs):
#    model_name = 'se_resnet101'
    submits = []
    types = [ 0  for i in range(6391)]
    file_names = [ ''  for i in range(6391)]
    model_setting = model_settings[model_name]

    if 'efficientnet' in model_name:
        model = EfficientNet_new.from_pretrained(model_name=model_name,num_classes=9,dropout_rate=model_setting['dropout'])
    elif 'resnext101_32x8d_wsl' in model_name:
        net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        model = Net(net,final_feature=model_setting['final_feature'],dropout=model_setting['dropout'])
    else:
        net = pretrainedmodels.__dict__[model_name](num_classes=1000)
        model = Net(net, final_feature=model_setting['final_feature'], dropout=model_setting['dropout'])
    model.cuda()
    model = nn.DataParallel(model)

    model_setting = model_settings[model_name]
    test_transforms = transforms.Compose([
            transforms.Resize(model_setting['resize_shape']),
            transforms.CenterCrop(model_setting['input_size']),
            transforms.ToTensor(),
            model_setting['transforms_norm']
        ])
    for i in range(5):
        test_index = k_folds[i][1]
        test_dataset = ProductDataset('Train_label.csv','train',test_index, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=model_setting['batch_size'],
                                                  shuffle=False,
                                                  **kwargs
                                                  )
        model.load_state_dict(torch.load('model/%s_%d_best.model' % (model_name, i)))

        testtypes= testY(model,test_loader)
        for index in range(len(test_index)):
            types[test_index[index]] = testtypes[index]
            file_names[test_index[index]] = test_dataset.file_name[index]

    dataframe = pd.DataFrame({'FileName': file_names, 'type': types})
    dataframe.to_csv('result/test_label_%s.csv'%model_name, index=False, sep=',')
    print('generator_label finish!')

def val(epoch, model,data_loader):
    print("\nValidation Epoch: %d" % epoch)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
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


def testY(model, test_loader):
    print("\nstart testY")
    model.eval()

    types = []
    softmax = nn.Softmax(dim=1).cuda()

    with torch.no_grad():
        all_output = None
        for _ in range(1):
            outputs = None
            for batch_idx, data in enumerate(test_loader):
                images, labels,_ = data
                if torch.cuda.is_available():
                    images = images.cuda()
                out = model(images)
                out = softmax(out).cpu().numpy()
                if outputs is not None:
                    outputs = np.concatenate((outputs, out))
                else:
                    outputs = out
            if all_output is not None:
                all_output += outputs
            else:
                all_output = outputs
        predicted = np.argmax(all_output, 1)
        types += [i + 1 for i in predicted.tolist()]
    return types

def test(model, test_loader,file_name_list, file_name=None,times=8):
    print("\nstart test")
    model.eval()

    types = []
    softmax = nn.Softmax(dim=1).cuda()

    with torch.no_grad():
        all_output = None
        for _ in range(times):
            outputs = None
            for batch_idx, data in enumerate(test_loader):
                images, _ = data
                if torch.cuda.is_available():
                    images = images.cuda()
                out = model(images)
                out = softmax(out).cpu().numpy()
                if outputs is not None:
                    outputs = np.concatenate((outputs, out))
                else:
                    outputs = out
            if all_output is not None:
                all_output += outputs
            else:
                all_output = outputs
        predicted = np.argmax(all_output, 1)
        types += [i + 1 for i in predicted.tolist()]
    dataframe = pd.DataFrame({'FileName': file_name_list, 'type': types})
    dataframe.to_csv(file_name, index=False, sep=',')

