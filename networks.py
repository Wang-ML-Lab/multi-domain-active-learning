'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable
import numpy as np

# mlp model class
class mlpE(nn.Module):
    def __init__(self, opts):
        super(mlpE, self).__init__()
        self.embSize = opts.nEmb
        self.domain_num = opts.domain_num
        self.dim = int(np.prod(opts.dim))
        self.lm1 = nn.Linear(self.dim+100, self.embSize)
        self.lm2 = nn.Linear(self.embSize, opts.class_num)
        # self.seed = opts.seed
        # self.save_name = opts.save_name
    def forward(self, x, dm_indx):
        dm_indx = dm_indx.view(-1, 1).repeat(1, 100)/(self.domain_num-1)
        x = x.view(-1, self.dim)
        x = torch.cat((dm_indx, x), dim=1)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb

    # def get_embedding_dim(self):
    #     return self.embSize
    # def save(self, rd):
    #     name = self.save_name+'_'+ str(self.seed) +'_'+ str(rd) +'.pth'
    #     torch.save(self.state_dict(), name)

    # def load(self, rd):
    #     name = self.save_name+'_'+ str(self.seed) +'_'+ str(rd) +'.pth'
    #     try:
    #         print('load model from {}'.format(name))
    #         self.load_state_dict(torch.load(name))
    #         print('done!')
    #     except:
    #         print('failed!')

class mlpD(nn.Module):
    def __init__(self, opts):
        super(mlpD, self).__init__()
        self.embSize = opts.nEmb
        self.lm1 = nn.Linear(self.embSize, self.embSize)
        self.lm2 = nn.Linear(self.embSize, opts.domain_num)
    def forward(self, emb):
        layer1 = F.relu(self.lm1(emb))
        out = torch.sigmoid(self.lm2(layer1))
        return out
###############
class EncoderSTN_no_alpha(nn.Module):
    def __init__(self, opt):
        super(EncoderSTN_no_alpha, self).__init__()

        nh = 256
        self.domain_num = opt.domain_num
        nz = opt.nz

        self.conv = nn.Sequential(
            nn.Conv2d(2, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 14 x 14
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 7 x 7
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 4 x 4
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nnSqueeze(),
            nn.Linear(nh, 10)
        )



    def forward(self, x, domain_index):
        """
        :param x: B x 1 x 28 x 28
        :param domain_index: B x 1
        :return:
        """
        # x = self.stn(x, u)
        domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 1, 28, 28)/(self.domain_num-1)
        x = torch.cat((x, domain_index), dim=1)
        z = self.conv(x)
        y = self.fc_pred(z)
        return y, z

class Dis_softmax(nn.Module):
    def __init__(self, opt):
        super(Dis_softmax, self).__init__()
        nh = 512
        nin=opt.nz
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, opt.domain_num+1)
        )

    def forward(self, x):
        return self.net(x)

class  DiscConv(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv, self).__init__()
        nh = 512
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, nout),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Dis_6d(nn.Module):
    def __init__(self, opt):
        super(Dis_6d, self).__init__()
        nh = 512
        nin= opt.nz
        self.nout = opt.domain_num
        self.net ={}
        self.net_para =[]
        for i in range(self.nout):
            self.net[i] = nn.Sequential(
                nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
                nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
                nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
                nnSqueeze(),
                nn.Linear(nh, 1),
                nn.Sigmoid()
            )
            self.net[i] = self.net[i].cuda()
            self.net_para += list(self.net[i].parameters())
    

    def forward(self, x):
        out = []
        for i in range(self.nout):
            out_i = self.net[i](x).view(-1, 1)
            out.append(out_i)
        return torch.cat(out, dim=1)
###############
# mlp model class
class mlpEc(nn.Module):
    def __init__(self, opts):
        super(mlpEc, self).__init__()
        self.embSize = opts.nEmb
        self.domain_num = opts.domain_num
        self.dim = int(np.prod(opts.dim))
        self.lm1 = nn.Linear(self.dim+100, 2*self.embSize)
        self.lm2 = nn.Linear(2*self.embSize, self.embSize)
        self.lm3 = nn.Linear(self.embSize, self.embSize)
        self.lm4 = nn.Linear(self.embSize, self.embSize)
        self.lm5 = nn.Linear(self.embSize, opts.class_num)
        # self.seed = opts.seed
        # self.save_name = opts.save_name
    def forward(self, x, dm_indx):
        dm_indx = dm_indx.view(-1, 1).repeat(1, 100)/(self.domain_num-1)
        x = x.view(-1, self.dim)
        x = torch.cat((dm_indx, x), dim=1)
        x = F.relu(self.lm1(x))
        x = F.relu(self.lm2(x))
        emb = F.relu(self.lm3(x))
        x = F.relu(self.lm4(emb))
        out = self.lm5(x)
        return out, emb

# mlp model class
class MNIST_mlpE(nn.Module):
    def __init__(self, opts):
        super(MNIST_mlpE, self).__init__()
        self.embSize = opts.nEmb
        self.domain_num = opts.domain_num
        self.dim = int(np.prod(opts.dim))
        self.lm1 = nn.Linear(self.dim+100, 2*self.embSize)
        self.lm2 = nn.Linear(2*self.embSize, self.embSize)
        self.lm3 = nn.Linear(self.embSize, self.embSize)
        self.lm4 = nn.Linear(self.embSize, self.embSize)
        self.lm5 = nn.Linear(self.embSize, opts.class_num*(self.domain_num+1))
        # self.seed = opts.seed
        # self.save_name = opts.save_name
    def forward(self, x, dm_indx):
        dm_indx = dm_indx.view(-1, 1).repeat(1, 100)/(self.domain_num-1)
        x = x.view(-1, self.dim)
        x = torch.cat((dm_indx, x), dim=1)
        x = F.relu(self.lm1(x))
        x = F.relu(self.lm2(x))
        emb = F.relu(self.lm3(x))
        x = F.relu(self.lm4(emb))
        out = self.lm5(x)
        return out, emb

class MNIST_mlpD(nn.Module):
    def __init__(self, opts):
        super(MNIST_mlpD, self).__init__()
        self.embSize = opts.nEmb
        self.lm1 = nn.Linear(self.embSize, self.embSize)
        self.lm2 = nn.Linear(self.embSize, self.embSize)
        self.lm3 = nn.Linear(self.embSize, opts.domain_num)
    def forward(self, emb):
        layer1 = F.relu(self.lm1(emb))
        layer2 = F.relu(self.lm2(layer1))
        out = torch.sigmoid(self.lm3(layer2))
        return out

class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)
#############

class MNIST_Encoder(nn.Module):
    def __init__(self, opt):
        super(MNIST_Encoder, self).__init__()

        nh = 256
        self.domain_num = opt.domain_num
        nz = opt.nz

        self.conv = nn.Sequential(
            nn.Conv2d(1, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 14 x 14
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 7 x 7
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 4 x 4
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nnSqueeze(),   
        )
        
        self.last_layer = nn.Linear(nh, 10*(opt.domain_num+1))
    def forward(self, x, domain_index):
        """
        :param x: B x 1 x 28 x 28
        :param domain_index: B x 1
        :return:
        """
        # x = self.stn(x, u)
        # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 1, 28, 28)/(self.domain_num-1)
        # x = torch.cat((x, domain_index), dim=1)
        z = self.conv(x)
        em = self.fc_pred(z)
        y = self.last_layer(em)
        return y, z
    
    def get_emb(self, x, domain_index):
        """
        :param x: B x 1 x 28 x 28
        :param domain_index: B x 1
        :return:
        """
        # x = self.stn(x, u)
        # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 1, 28, 28)/(self.domain_num-1)
        # x = torch.cat((x, domain_index), dim=1)
        z = self.conv(x)
        em = self.fc_pred(z)
        y = self.last_layer(em)
        return y, em
    def get_parameters(self):
        return self.parameters()

class MNIST_Dis(nn.Module):
    def __init__(self, opt):
        super(MNIST_Dis, self).__init__()
        """Input: features condition on domain index"""
        nh = 512
        self.domain_num = opt.domain_num
        self.dm_d = 10 
        nin = opt.nz+self.dm_d
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, 1),
            nn.Sigmoid()
        )

    def forward(self, x, domain_index):
        """
        :param x: B x 1 x 28 x 28
        :param domain_index: B x 1
        :return:
        """
        # x = self.stn(x, u)
        b, c, h, w = x.size()
        domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, self.dm_d, h, w)/(self.domain_num-1)
        x = torch.cat((x, domain_index), dim=1)
        return self.net(x)
    def get_parameters(self):
        return self.parameters()

class MNIST_Dis_1d(nn.Module):
    def __init__(self, opt):
        super(MNIST_Dis_1d, self).__init__()
        """Input: features condition on domain index"""
        nh = 512
        self.domain_num = opt.domain_num
        nin = opt.nz
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: B x 1 x 28 x 28
        :param domain_index: B x 1
        :return:
        """
        return self.net(x)
    def get_parameters(self):
        return self.parameters()
