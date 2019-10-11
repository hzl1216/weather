import torch
import torch.nn as nn
import numpy as np
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
