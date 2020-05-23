import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time


def train_one_epoch(model, optimizer, criterion, train_loader, epoch, max_epoch, disp_freq, cuda, print_true):
    batch_train_loss, batch_train_acc = [], []
    model.train()
    for i, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        predictions, loss = model(data, target, criterion)

        pred = predictions.data.max(1)[1]  # get the index of the max log-probability
        acc = pred.eq(target.data).sum().item() / len(target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_train_loss.append(loss.item())
        batch_train_acc.append(acc)
        if (i+1) % disp_freq == 0 and print_true == True:
            print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                epoch, max_epoch, i+1, len(train_loader),
                loss.item(), acc))
    return batch_train_loss, batch_train_acc


def validate(model, criterion, val_iter, cuda):
    batch_val_acc, batch_val_loss = [], []
    model.eval()
    for i, (data, target) in enumerate(val_iter):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        predictions, loss = model(data, target, criterion)

        pred = predictions.data.max(1)[1]  # get the index of the max log-probability
        acc = pred.eq(target.data).sum().item() / len(target)

        batch_val_loss.append(loss.item())
        batch_val_acc.append(acc)
    return batch_val_loss, batch_val_acc


def test_model(model, criterion, test_iter, cuda):
    print('Testing...')

    batch_test_acc, batch_test_loss = [], []
    model.eval()
    for i, (data, target) in enumerate(test_iter):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        predictions, loss = model(data, target, criterion)

        pred = predictions.data.max(1)[1]  # get the index of the max log-probability
        acc = pred.eq(target.data).sum().item() / len(target)

        batch_test_loss.append(loss.item())
        batch_test_acc.append(acc)

    print("The test accuracy is {:.4f}.\n".format(np.mean(batch_test_acc)))
    return np.mean(batch_test_acc)


def train_val(model, optimizer, criterion, train_iter, val_iter, max_epoch, disp_freq, cuda, print_true):
    avg_train_loss, avg_train_acc = [], []
    avg_val_loss, avg_val_acc = [], []
    #start = time.time()

    for epoch in range(1, max_epoch+1):
        batch_train_loss, batch_train_acc = train_one_epoch(model, optimizer, criterion, train_iter,
                                                            epoch, max_epoch, disp_freq, cuda, print_true)
        batch_val_loss, batch_val_acc = validate(model, criterion, val_iter, cuda)

        avg_train_acc.append(np.mean(batch_train_acc))
        avg_train_loss.append(np.mean(batch_train_loss))
        avg_val_acc.append(np.mean(batch_val_acc))
        avg_val_loss.append(np.mean(batch_val_loss))
        if print_true == True:
            print()
            print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
                epoch, avg_train_loss[-1], avg_train_acc[-1]))

            print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}'.format(
                epoch, avg_val_loss[-1], avg_val_acc[-1]))
            print()
    #end = time.time()
    #print('Training time is {:.4f}s'.format(end-start))
    return model, avg_val_loss, avg_val_acc#, (end-start)


def plot_loss_and_acc(loss_and_acc_dict):
    plt.figure()
    plt.subplot(211)
    tmp = list(loss_and_acc_dict.values())
    maxEpoch = len(tmp[0][0])
    stride = np.ceil(maxEpoch / 10)

    maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
    minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)

    for name, lossAndAcc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + maxEpoch), lossAndAcc[0], '-s', label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(range(0, maxEpoch + 1, 2))
    plt.axis([0, maxEpoch, minLoss, maxLoss])


    maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
    minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)

    plt.subplot(212)
    for name, lossAndAcc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(range(0, maxEpoch + 1, 2))
    plt.axis([0, maxEpoch, minAcc, maxAcc])
    plt.legend()
    plt.show()