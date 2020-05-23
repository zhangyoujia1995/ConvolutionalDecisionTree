import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict


class InnerNode():
    def __init__(self, depth, input_channel, input_height, args):
        self.args = args

        self.conv = nn.Sequential(OrderedDict([
            ('convolution', nn.Conv2d(input_channel, 8*2**(depth-1), kernel_size=(3, 3))),
            ('batch_normalization', nn.BatchNorm2d(8 * 2 ** (depth - 1))),
            ('relu', nn.ReLU()),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        fc_inputshape = self.get_outputshape(input_channel, input_height)
        self.fc = nn.Linear(fc_inputshape[-1] * fc_inputshape[-2] * fc_inputshape[-3], 1)

        beta = torch.randn(1)
        if self.args.cuda:
            beta = beta.cuda()
        self.beta = nn.Parameter(beta)
        self.leaf = False
        self.prob = None
        self.leaf_accumulator = []
        self.lmbda = self.args.lmbda * 2 ** (-depth)
        self.build_child(depth, input_channel, input_height)
        self.penalties = []

    def reset(self):
        self.leaf_accumulator = []
        self.penalties = []
        self.left.reset()
        self.right.reset()

    def get_outputshape(self, input_channel, input_height):
        x = Variable(torch.randn(1, input_channel, input_height, input_height).type(torch.FloatTensor),
                     requires_grad=False)
        return self.conv(x).size()

    def build_child(self, depth, input_channel, input_height):
        fc_inputshape = self.get_outputshape(input_channel, input_height)
        if depth < self.args.max_depth:
            self.left = InnerNode(depth + 1, fc_inputshape[1], fc_inputshape[-1], self.args)
            self.right = InnerNode(depth + 1, fc_inputshape[1], fc_inputshape[-1], self.args)
        else:
            self.left = LeafNode(fc_inputshape[1], fc_inputshape[-1], self.args)
            self.right = LeafNode(fc_inputshape[1], fc_inputshape[-1], self.args)

    def cal_prob(self, x, path_prob):
        x = self.conv(x)
        out = self.fc(x.reshape(x.shape[0], -1))

        self.prob = F.sigmoid(self.beta * out)  # probability of selecting right node

        self.path_prob = path_prob

        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1 - self.prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return self.leaf_accumulator

    def get_penalty(self):
        penalty = (torch.sum(self.prob * self.path_prob) / torch.sum(self.path_prob), self.lmbda)
        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return self.penalties


class LeafNode():
    def __init__(self, input_channel, input_height, args):
        self.args = args
        self.leaf = True
        self.fc_leaf = nn.Linear(input_channel * input_height * input_height, 10)
        self.softmax = nn.Softmax()

    def reset(self):
        pass

    def cal_prob(self, x, path_prob):
        out = self.fc_leaf(x.reshape(x.shape[0], -1))
        Q = out
        Q = Q.expand((self.args.batch_size, self.args.output_dim))
        return [[path_prob, Q]]


class CDTree(nn.Module):  # Convolution Decision Tree
    def __init__(self, args):
        super(CDTree, self).__init__()
        self.args = args
        self.root = InnerNode(1, args.input_nc, args.input_height, self.args)
        self.collect_parameters()
        self.test_acc = []
        self.best_accuracy = 0.0
        self.path_prob_init = Variable(torch.ones(self.args.batch_size, 1)/2)
        self.init_depth = 0
        if self.args.cuda:
            self.path_prob_init = self.path_prob_init.cuda()

    def forward(self, x, y, criterion):
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(self.args.batch_size)]
        max_Q = [torch.zeros(self.args.output_dim) for _ in range(self.args.batch_size)]
        for (path_prob, Q) in leaf_accumulator:
            loss_leaf = criterion(Q, y)
            loss += path_prob * loss_leaf
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            path_prob_numpy = np.nan_to_num(path_prob_numpy)
            for i in range(self.args.batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 * (torch.log(penalty) + torch.log(1 - penalty))
        output = torch.stack(max_Q)
        self.root.reset()  ##reset all stacked calculation
        return output, loss+C+0.001 ## -log(loss) will always output non, because loss is always below zero. I suspect this is the mistake of the paper?

    def collect_parameters(self):
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        while nodes:
            node = nodes.pop(0)
            if node.leaf:
                fc2 = node.fc_leaf
                self.module_list.append(fc2)
            else:
                fc1 = node.fc
                conv1 = node.conv

                beta = node.beta
                nodes.append(node.right)
                nodes.append(node.left)

                self.param_list.append(beta)
                self.module_list.append(conv1)
                self.module_list.append(fc1)


class CDForest(nn.Module):
    def __init__(self, args):
        super(CDForest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = args.n_tree
        self.weight = torch.randn(self.n_tree)
        self.weight = nn.Parameter(self.weight, requires_grad=True)
        for _ in range(self.n_tree):
            tree = CDTree(args)
            self.trees.append(tree)

    def forward(self, x, y):
        probs, losses = [], []
        for i, tree in enumerate(self.trees):
            prob, loss = tree(torch.rot90(x, i-1, [2, 3]), y)
            probs.append(prob.unsqueeze(2))
            losses.append(loss.unsqueeze(0))
        losses = torch.cat(losses, dim=0)
        probs = torch.cat(probs, dim=2)
        normalized_weights = nn.functional.softmax(self.weight, dim=-1)
        probs = nn.functional.linear(probs, normalized_weights)
        losses = nn.functional.linear(losses, normalized_weights)
        return probs, losses