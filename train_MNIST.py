from dl import cdt, solver
import torch.nn as nn
import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Convolution Decision Tree Example(MNIST)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--input-nc', type=int, default=1, metavar='N',
                    help='input number of channel(default: 1)')
parser.add_argument('--input-dim', type=int, default=28*28, metavar='N',
                    help='input dimension size(default: 28*28)')
parser.add_argument('--input-height', type=int, default=28, metavar='N',
                    help='input height size(default: 28)')
parser.add_argument('--output-dim', type=int, default=10, metavar='N',
                    help='output dimension size(default: 10)')
parser.add_argument('--max-depth', type=int, default=3, metavar='N',
                    help='maximum depth of tree(default: 3)')
parser.add_argument('--lmbda', type=float, default=0.01, metavar='LR',
                    help='temperature rate (default: 0.01)')
parser.add_argument('--n-tree', type=int, default=3, metavar='N',
                    help='number of trees for CDForest(default: 3)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--max-epoch', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--disp-freq', type=int, default=100, metavar='N',
                    help='display frequency')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--mode', default='client')
parser.add_argument('--port',default=64284)
args = parser.parse_args()


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(1)

    # load mnist data
    train_dataset = datasets.MNIST(root='./MNIST/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./MNIST/', train=False, transform=transforms.ToTensor(), download=True)
    # data preprocessing
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    #model
    model = cdt.CDTree(args)
    # for forest
    # model = cdt.CDForest(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
        print('RUNNING WITH GPU')
    else:
        model.cpu()
        print("WARNING: RUNNING WITHOUT GPU")

    print('Training...')
    print('Total epochs: {}'.format(args.max_epoch))
    print('Batch size: {}'.format(args.batch_size))
    model, train_loss, train_acc = solver.train_val(model, optimizer, criterion, train_loader, val_loader,
                                                    args.max_epoch, args.disp_freq, args.cuda, True)

    test_acc = solver.test_model(model, criterion, test_loader, args.cuda)
    solver.plot_loss_and_acc({'CDTree': [train_loss, train_acc]})

if __name__ == '__main__':
    main()

