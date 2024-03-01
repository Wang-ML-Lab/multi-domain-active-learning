import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import math
import torch.nn.functional as F




######
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)

########resnet feature

class DN_resnet(nn.Module):
    def __init__(self, resnet_name):
        super(DN_resnet, self).__init__()
        
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool) # N*512*7*7
    def forward(self, x):
        return self.feature_layers(x) # extract feature


########2

class DN_Encoder_resnet(nn.Module):
    def __init__(self, opts):
        super(DN_Encoder_resnet, self).__init__()
        resnet_name, nz, class_num \
        = opts.resnet_name, opts.nz, opts.class_num
        self.domain_num = opts.domain_num
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool) # N*512*7*7 #

        nh= 512
        self.resnet_name = opts.resnet_name
        if opts.resnet_name == "ResNet50":
            c_num=2
        else:
            c_num=1
        self.conv = nn.Sequential(
            nn.Conv2d(c_num, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 16 x 16
            nn.Conv2d(nh, nh, 3, 2, 0), nn.BatchNorm2d(nh), nn.ReLU(True),  # 7 x 7
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 4 x 4
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nnSqueeze(),   
        )
        
        self.last_layer = nn.Linear(nh, class_num)

    def forward(self, x, domain_index):
        x = self.feature_layers(x)
        if self.resnet_name == "ResNet50":
            x = x.view(-1, 2, 32, 32) # resnet 50
            # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 2, 32, 32)/(self.domain_num-1)
        else:
            x = x.repeat(1, 2, 1, 1).view(-1, 1, 32, 32) # resnet 18
            # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 1, 32, 32)/(self.domain_num-1)
        # x = torch.cat((x, domain_index), dim=1)
      
        z = self.conv(x)
        em = self.fc_pred(z)
        y = self.last_layer(em)

        return y, z

    def get_emb(self, x, domain_index):
        x = self.feature_layers(x)      
        if self.resnet_name == "ResNet50":
            x = x.view(-1, 2, 32, 32) # resnet 50
            # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 2, 32, 32)/(self.domain_num-1)
        else:
            x = x.repeat(1, 2, 1, 1).view(-1, 1, 32, 32) # resnet 18
            # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 1, 32, 32)/(self.domain_num-1)

        
        # x = torch.cat((x, domain_index), dim=1)
        z = self.conv(x)
        em = self.fc_pred(z)
        y = self.last_layer(em)
        return y, em 
    #     nh= 256
    #     if resnet_name == "ResNet50":
    #         input_dim = 2048
    #     else:
    #         input_dim = 512
    #     self.conv = nn.Sequential(
    #         nn.Conv2d(input_dim, nh, 3, 1, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 7 x 7
    #         nn.Conv2d(nh, nh, 3, 1, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 7 x 7
    #         nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 4 x 4
    #         nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
    #     )

    #     self.fc_pred = nn.Sequential(
    #         nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
    #         nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
    #         nnSqueeze(),   
    #     )
        
    #     self.last_layer = nn.Linear(nh, class_num)

    #     self.__in_features = nz


    # def forward(self, x, domain_index):
    #     x = self.feature_layers(x) # extract feature

    #     # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 512, 7, 7)/(self.domain_num-1)
    #     # x = torch.cat((x, domain_index), dim=1)
        
    #     z = self.conv(x)
    #     em = self.fc_pred(z)
    #     y = self.last_layer(em)
    #     return y, z

    # def get_emb(self, x, domain_index):
        # x = self.feature_layers(x) # extract feature

        # # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 512, 7, 7)/(self.domain_num-1)
        # # x = torch.cat((x, domain_index), dim=1)
        # z = self.conv(x)
        # em = self.fc_pred(z)
        # y = self.last_layer(em)
        # return y, em 

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        
        parameter_list = list(self.feature_layers.parameters()) + list(self.conv.parameters()) +list(self.fc_pred.parameters()) + list(self.last_layer.parameters()) #

        return parameter_list

class cls(nn.Module):
    def __init__(self, opts):
        super(cls, self).__init__()
        nz, class_num \
        = opts.nz, opts.class_num
        self.domain_num = opts.domain_num

        nh= 256
        self.conv = nn.Sequential(
            nn.Conv2d(3, nh, 7, 4, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 55 x 55
            nn.Conv2d(nh, nh, 5, 3, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 18 x 18
            nn.Conv2d(nh, nh, 5, 3, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 6 x 6
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 3 x 3
            nn.Conv2d(nh, nz, 3, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nnSqueeze(),   
        )
        
        self.last_layer = nn.Linear(nh, class_num)

        self.__in_features = nz


    def forward(self, x, domain_index):

        # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 512, 7, 7)/(self.domain_num-1)
        # x = torch.cat((x, domain_index), dim=1)
        
        z = self.conv(x)
        em = self.fc_pred(z)
        y = self.last_layer(em)
        return y, z

    def get_emb(self, x, domain_index):

        # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 512, 7, 7)/(self.domain_num-1)
        # x = torch.cat((x, domain_index), dim=1)
        z = self.conv(x)
        em = self.fc_pred(z)
        y = self.last_layer(em)
        return y, em 

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        
        parameter_list = list(self.conv.parameters()) +list(self.fc_pred.parameters()) + list(self.last_layer.parameters()) #

        return parameter_list


class DN_Encoder(nn.Module):
    def __init__(self, opts):
        super(DN_Encoder, self).__init__()
        """feature version"""
        nz,class_num \
        =  opts.nz, opts.class_num*(opts.domain_num+1)
        self.domain_num = opts.domain_num
        if opts.nh:
            nh= opts.nh
        else:
            nh = 256
        self.resnet_name = opts.resnet_name
        if opts.resnet_name == "ResNet50":
            c_num=2
        else:
            c_num=1
        self.conv = nn.Sequential(
            nn.Conv2d(c_num, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 16 x 16
            nn.Conv2d(nh, nh, 3, 2, 0), nn.BatchNorm2d(nh), nn.ReLU(True),  # 7 x 7
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 4 x 4
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nnSqueeze(),   
        )
        
        self.last_layer = nn.Linear(nh, class_num)

    def forward(self, x, domain_index):

        if self.resnet_name == "ResNet50":
            x = x.view(-1, 2, 32, 32) # resnet 50
            # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 2, 32, 32)/(self.domain_num-1)
        else:
            x = x.repeat(1, 2, 1, 1).view(-1, 1, 32, 32) # resnet 18
            # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 1, 32, 32)/(self.domain_num-1)
        # x = torch.cat((x, domain_index), dim=1)
        z = self.conv(x)
        # print(z.shape)
        em = self.fc_pred(z)
        y = self.last_layer(em)
        return y, z

    def get_emb(self, x, domain_index):

        if self.resnet_name == "ResNet50":
            x = x.view(-1, 2, 32, 32) # resnet 50
            # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 2, 32, 32)/(self.domain_num-1)
        else:
            x = x.repeat(1, 2, 1, 1).view(-1, 1, 32, 32) # resnet 18
            # domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, 1, 32, 32)/(self.domain_num-1)
        # x = torch.cat((x, domain_index), dim=1)
        z = self.conv(x)
        em = self.fc_pred(z)
        y = self.last_layer(em)
        return y, em 


    def get_parameters(self):

        parameter_list = list(self.conv.parameters()) \
                    + list(self.fc_pred.parameters()) + list(self.last_layer.parameters())
        return parameter_list

class DN_Dis(nn.Module):
    def __init__(self, opt):
        super(DN_Dis, self).__init__()
        """Input: features condition on domain index"""
        nh = 512
        self.domain_num = opt.domain_num
        self.dm_d = 10 
        nin = opt.nz+self.dm_d
        # ### old 2023-03-19
        # self.net = nn.Sequential(
        #     nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
        #     nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
        #     nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
        #     nnSqueeze(),
        #     nn.Linear(nh, 1),
        #     nn.Sigmoid()
        # )
        #  ### old 2023-03-19

        ### new 2023-03-19
        self.net1 = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            
        )
        self.net2 = nn.Sequential(
            nn.Linear(nh, 1),
            nn.Sigmoid()
        )
         ### new 2023-03-19

    def forward(self, x, domain_index):
        """
        :param x: B x 1 x 28 x 28
        :param domain_index: B x 1
        :return:
        """
        ### mode#1: before 2022-5-24
        # x = self.stn(x, u)
        b, c, h, w = x.size()
        domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, self.dm_d, h, w)/(self.domain_num-1)

        x = torch.cat((x, domain_index), dim=1)
        # return self.net(x)
        # print("forward", x.shape)
        return self.net2(self.net1(x))
    def get_parameters(self):
        return self.parameters()
    
    def get_emb(self, x, domain_index):
        """
        :param x: B x 1 x 28 x 28
        :param domain_index: B x 1
        :return:
        """
        ### mode#1: before 2022-5-24
        # x = self.stn(x, u)
        b, c, h, w = x.size()
        domain_index = domain_index.view(-1, 1, 1, 1).repeat(1, self.dm_d, h, w)/(self.domain_num-1)

        x = torch.cat((x, domain_index), dim=1)
        # return self.net(x)
        # print("get_emb", x.shape)
        return self.net2(self.net1(x)), self.net1(x)


class DN_Dis_one_hot(nn.Module):
    def __init__(self, opt):
        super(DN_Dis_one_hot, self).__init__()
        """Input: features condition on domain index"""
        nh = 512
        self.domain_num = opt.domain_num
        nin = opt.nz+self.domain_num
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
        ### mode#2: in 2022-5-24
        # x = self.stn(x, u)
        b, c, h, w = x.size()
        domain_index = F.one_hot(domain_index.long(), num_classes = self.domain_num)
        domain_index = domain_index.view(-1, self.domain_num, 1, 1).repeat(1, 1, h, w)

        x = torch.cat((x, domain_index), dim=1)
        return self.net(x)
    def get_parameters(self):
        return self.parameters()

class DN_Dis_one_hot_woBN(nn.Module):
    def __init__(self, opt):
        super(DN_Dis_one_hot_woBN, self).__init__()
        """Input: features condition on domain index"""
        nh = 512
        self.domain_num = opt.domain_num
        nin = opt.nz+self.domain_num
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.LeakyReLU(), 
            nn.Conv2d(nh, nh, 1, 1, 0), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.LeakyReLU(),
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
        ### mode#2: in 2022-5-24
        # x = self.stn(x, u)
        b, c, h, w = x.size()
        domain_index = F.one_hot(domain_index.long(), num_classes = self.domain_num)
        domain_index = domain_index.view(-1, self.domain_num, 1, 1).repeat(1, 1, h, w)

        x = torch.cat((x, domain_index), dim=1)
        return self.net(x)
    def get_parameters(self):
        return self.parameters()


class DN_Dis_1d(nn.Module):
    def __init__(self, opt):
        super(DN_Dis_1d, self).__init__()
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


class DN_Dis_DANN(nn.Module):
    def __init__(self, opt):
        super(DN_Dis_DANN, self).__init__()
        """Input: features condition on domain index"""
        nh = 512
        self.domain_num = opt.domain_num
        self.dm_d = 0 
        nin = opt.nz+self.dm_d
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, opt.domain_num),
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

