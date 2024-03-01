# from json import encoder
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import os
# from progressbar import ProgressBar
import torch.optim.lr_scheduler as lr_scheduler
from prettytable import PrettyTable
from PIL import Image
from matplotlib.cm import get_cmap
from networks import *
import torch.optim as optim
from torch.autograd import Function
from nn_resnets import *

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
def get_net():
    return alphaDA

def EntropyLoss(input_): # others'
    mask = input_.ge(1e-6)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))
# ======================================================================================================================
def to_np(x):
    return x.detach().cpu().numpy()
class BaseModel(nn.Module):
    def __init__(self, opts):
        super(BaseModel, self).__init__()
        self.opt = opts
        self.domain_num = opts.domain_num
        self.seed = opts.seed
        self.save_name = opts.save_name
        self.lr = opts.lr
        self.tsne = TSNE(n_components=2)
        self.pca = PCA(n_components=2)
        self.softmax = nn.Softmax(dim=1)
        # self.train_log = self.save_name+'_'+ str(self.seed)+ '_train.log'

        if opts.data == 'MNIST':
            self.discriminator = MNIST_Dis
            encoder = MNIST_Encoder
            self.T = 10.0
        elif opts.data in ['domainnet', 'office-home', 'imageCLEF', 'office31', 'office_caltech','cct20']:
            self.discriminator = DN_Dis
            encoder = DN_Encoder
            self.T = opts.T
            
        self.netD = self.discriminator(self.opt).cuda()
        self.netE = encoder(self.opt).cuda()
        self.encoder = encoder


        self.lambda_gan = opts.lambda_gan
        self.lambda_A = opts.lambda_A

        if opts.third_term_ablation == True:
            self.lambda_3rd_term = 0
        else:
            self.lambda_3rd_term = 1
        
        self.adv_loss = torch.nn.BCELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)
        self.beta_tr = self.opt.beta.cuda() # training set
        self.beta = (torch.ones(opts.domain_num)/opts.domain_num).cuda() # this is right because self.beta will be given value in run.py 
        # self.beta = self.opt.beta.cuda() # training labeled set
        self.eye = torch.eye(self.opt.domain_num).cuda()

        self.logger_train = opts.logger_train
        self.logger_final = opts.logger_final
        self.save_dict = {}

    def reset_nn4domainnet(self):
        self.netD = self.discriminator(self.opt).cuda()
        self.netE = self.encoder(self.opt).cuda()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            # print(net)
            if net is not None:
                # print(net.get_parameters())
                for param in net.get_parameters():
                    param.requires_grad = requires_grad
    def set_input(self, input):
        self.x, self.y, self.dm = input

    def forward(self):
        # print(self.x.type(), self.dm.type())
        self.out, self.e = self.netE(self.x, self.dm)
        self.pred = torch.argmax(self.out[:, :self.opt.class_num].detach(), dim=1)

    def save(self):
        name = self.save_name+'_'+ str(self.seed) +'_'+ str(self.rd) +'.pth'
        torch.save(self.state_dict(), name)
        self.logger_train.info('save model in {}'.format(name))
    
    def load(self):
        name = self.save_name+'_'+ str(self.seed) +'_'+ str(self.rd) +'.pth'
        self.logger_train.info('load model from {}'.format(name))
        try:
            self.load_state_dict(torch.load(name))
            self.logger_train.info('done!')
        except:
            self.logger_train.info('failed!')
    
    def reset_net(self, net):
        def weight_reset(m):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.reset_parameters()
        return net.apply(weight_reset).cuda()
    
    def acc_reset(self):
        self.hit_domain, self.cnt_domain = np.zeros(self.opt.domain_num), np.zeros(self.opt.domain_num)
        self.cnt = 0
        self.hit = 0
    
    def acc_update(self):
        if self.pred.shape[0]==0:
            pass
        else:
            Y = to_np(self.y)
            P = to_np(self.pred)
            D = to_np(self.dm)
            D = (D).astype(np.int32)
            # print('shape',(Y == P).shape)
            hit = (Y == P).astype(np.float32)

            for i in range(self.opt.domain_num):
                self.hit_domain[i] += hit[D == i].sum()
                self.cnt_domain[i] += (D == i).sum()
    
    def acc_calc(self):
        self.acc_domain = self.hit_domain / self.cnt_domain
        self.acc = np.round(self.acc_domain.mean(), decimals=5)
        # mnist results: self.acc = np.round((self.acc_domain * self.opt.beta.cpu().numpy()).sum(), decimals=5)

    def print_log(self):
        self.logger_train.info(self.acc_msg)
        self.logger_train.info(self.loss_msg)

    def gen_result_table(self, dataloader):
        """print result table and save it as npy file"""
        res = PrettyTable()

        res.field_names = ["Accuracy"] + [f"Domain #{i}" for i in range(0, self.opt.domain_num)]

        hit = np.zeros((self.opt.class_num, self.opt.domain_num))
        cnt = np.zeros((self.opt.class_num, self.opt.domain_num))
        acc_table = np.zeros((self.opt.class_num+1, self.opt.domain_num+1))
        self.load()
        self.eval()
        with torch.no_grad():
            for x, y, dm, _, idxs in dataloader:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                self.set_input(input=(x, y, dm))
                self.forward()

                Y = to_np(self.y)
                G = to_np(self.pred)
                T = (to_np(self.dm)).astype(np.int32)

                for label, pred, domain in zip(Y, G, T):
                    # self.logger_final.info([label, pred, domain])
                    hit[label, domain] += int(label == pred)
                    # self.logger_final.info(int(label == pred))
                    cnt[label, domain] += 1
        cnt[cnt==0] +=1e-10
        for c in range(self.opt.class_num):
            res.add_row([f"Class {c}"] + list(np.round(100 * hit[c] / cnt[c], decimals=3)))

        res.add_row([f"Round{self.rd}'s Total"] + list(np.round(100 * hit.sum(0) / cnt.sum(0), decimals=3)))
        total_acc = (hit.sum(0) / cnt.sum(0) * self.opt.beta.cpu().numpy()).sum() # weighted acc according to training set
        avg_acc = (hit.sum(0) / cnt.sum(0)).mean() # average acc among all domains
        # final_acc_output = 'Total acc of %d-th round: '%self.rd+'%f'%total_acc + '\n'# mnist
        final_acc_output = 'Avg/total acc of %d-th round: '%self.rd+'%f'%avg_acc+'/%f'%total_acc + '\n'
        self.logger_final.info(res)
        self.logger_final.info(final_acc_output)
        acc_table[:self.opt.class_num, :self.opt.domain_num] = hit/cnt
        acc_table[self.opt.class_num, :self.opt.domain_num] =  hit.sum(0) / cnt.sum(0)
        acc_table[self.opt.class_num, self.opt.domain_num] =  avg_acc
        acc_table[self.opt.class_num-1, self.opt.domain_num] = total_acc # when mnist, exchange avg_acc and total_acc
        self.save_dict[self.rd]=acc_table
    
    def train_process(self, max_epoch, rd, train_set, train_label_set, test_set):
        self.rd = rd
        self.set_training()
        self.best_acc_test=0.0
        if self.opt.data in ['MNIST', 'domainnet', 'office-home', 'imageCLEF', 'office31', 'office_caltech','cct20']:
            epoch_set = test_set
            result_set = test_set
        elif self.opt.data=='domainnet':
            val_set, te_set = test_set
            epoch_set = val_set
            result_set = te_set

        for self.epoch in range(max_epoch):
            # print(self.epoch)
            self.learn(train_set, train_label_set)
            self.test(epoch_set)
        self.gen_result_table(result_set)
        self.save_final_result()
    
    def save_final_result(self):
        save_name = self.opt.save_name+'_'+ str(self.opt.seed)+ '_result.npy'
        np.save(save_name,self.save_dict)
    
        # read_dict = np.load(save_name).item() # load
    

class alphaDA(BaseModel):
    def __init__(self, opts):
        super(alphaDA, self).__init__(opts)
        if 'alpha' in self.opt.model:
            self.loss_names = ['D', 'A_gan', 'A_pred', 'E_gan', 'E_pred'] 
            # "D": discriminator loss, 'A_gan': domain gap loss to optimize alpha (see Sec. A2 in Append), 'A_pred': 1 - classfication accuracy
            #, 'E_gan': encoder's adversarial loss, 'E_pred': prediction loss
        else:
            self.loss_names = ['D', 'E_gan', 'E_pred']

    def save_npy(self, name, dic):
        save_name = self.opt.save_name+'_'+ str(self.opt.seed)+ '_'+ name +'.npy'
        np.save(save_name, dic)
    def load_npy(self, name):
        save_name = self.opt.save_name+'_'+ str(self.opt.seed)+ '_'+ name +'.npy'
        load_dict = np.load(save_name, allow_pickle=True).item()
        return load_dict
    def set_para(self):
        
        # self.alpha = nn.Parameter(torch.ones(self.opt.domain_num, self.opt.domain_num).cuda())
        
        if self.opt.alignment_ablation==True: # align labeled data and all data in each single domain
            self.alpha = nn.Parameter(torch.eye(self.opt.domain_num).cuda())
        else:
            self.alpha = nn.Parameter(torch.ones(self.opt.domain_num, self.opt.domain_num).cuda()) # set init alpha value

    def save_para(self):
        para_dict={}
        para_dict['alpha'] = self.alpha
        para_dict['mu'] = self.mu.detach().cpu().numpy()
        self.save_npy(str(self.rd), para_dict)  
    def load_para(self, name):
        return self.load_npy(str(self.rd-1))[name]
    def set_tr_im(self):
        """set whole training domains' importance weight
        beta_tr: train set ratio of every domain"""
        d_im = self.beta_tr #.view(1, -1).repeat(self.opt.domain_num, 1)
        d_im = 1/(self.opt.domain_num)/d_im
        self.tr_im = d_im

    def set_labeled_im(self):
        """set labeled domain importance weight
        beta: train set with labels ratio of every domain"""
        d_im = self.beta.view(1, -1).repeat(self.opt.domain_num, 1)
        d_im = 1/(self.opt.domain_num)/d_im
        # d_im[self.eye==1]=d_im[self.eye==1]*(self.opt.domain_num-1)
        self.labeled_im = d_im # 1/(K-1)/beta_j for combined domain and its corresponding domain in alignment and min(e_i, e_-i)
    def set_alpha(self):
        self.set_tr_im()
        self.set_labeled_im()
        
        self.big_alpha = self.softmax(self.alpha*self.T)
        self.sum_alpha = self.big_alpha.sum(dim=0)
        self.labeled_alpha = self.big_alpha*self.labeled_im
        self.mu = self.sum_alpha/self.sum_alpha.sum()
    def set_training(self):
        self.set_para()

        self.set_alpha()
        if self.opt.data =='MNIST':
            self.netE =  self.reset_net(self.netE)
            self.netD =  self.reset_net(self.netD)
            self.optimizer_G = torch.optim.Adam(self.netE.get_parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=self.opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.netD.get_parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=self.opt.weight_decay)
            self.optimizer_A = torch.optim.Adam([self.alpha], lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=self.opt.weight_decay)
        
        elif self.opt.data in ['office-home', 'imageCLEF', 'office_caltech']:
            self.reset_nn4domainnet()
            self.optimizer_G = torch.optim.Adam(self.netE.get_parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=self.opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.netD.get_parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=self.opt.weight_decay)
            self.optimizer_A = torch.optim.Adam([self.alpha], lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=self.opt.weight_decay)
        
    def learn(self, train_set, dataloader):
        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        self.acc_reset()
        len_train_set = len(train_set)
        len_label_set = len(dataloader)
        
        iter_label = iter(dataloader)
        iter_train = iter(train_set)
        
        if "alpha" in self.opt.model:
            if self.opt.alpha_update=='acc':
                opt_func = self.optimize_parameters_acc_alphaDA
            elif self.opt.alpha_update=='prob':
                opt_func = self.optimize_parameters_prob_alphaDA
            else:
                opt_func = self.optimize_parameters_alphaDA
        else:
            opt_func = self.optimize_parameters

        for i in range(len_label_set):
            if i % len_train_set == 0:
                iter_train = iter(train_set)
            if i % len_label_set == 0:
                iter_label = iter(dataloader)
            x, y, dm, _, _ = iter_label.next()
            x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
            xt, yt, dmt, _, _ = iter_train.next()
            xt, yt, dmt = Variable(xt.cuda()), Variable(yt.cuda()), Variable(dmt.cuda())
            self.set_input_train(input=(x, y, dm, xt, yt, dmt))
            opt_func()
            # print(self.big_alpha)
            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())
            
            self.acc_update()
        self.acc_calc()
        self.loss_msg = '[Round][{}][Train][{}] Loss:'.format( self.rd, self.epoch)
        for loss in self.loss_names:
            self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))
        self.acc_msg = '[Round][{}][Train][{}] Acc: {:.5f} ({}) ({}/{}) '.format(
            self.rd, self.epoch, self.acc, self.acc_domain, self.hit_domain, self.cnt_domain)
        self.print_log()
        # print('learn finish!')
    def set_input_train(self, input):
        self.x, self.y, self.dm, self.xt, self.yt,self.dmt = input
    def forward_train(self):
        self.out, self.e = self.netE(self.x, self.dm)
        self.pred = torch.argmax(self.out[:, :self.opt.class_num].detach(), dim=1)

        self.outt, self.et = self.netE(self.xt, self.dmt)
    
    def backward_acc_A(self):
        """
        fake: labeled domain, groundtruth is one
        real: orginal domain, groundtruth is zero
        """
        self.set_alpha()
        b = self.e.size(0)
        repeat_shape = [self.domain_num]+ [1 for i in range(len(self.e.shape)-1)] # if len(self.e.shape)=2, [domain_num, 1]
        # if len(self.e.shape)=4, [domain_num, 1, 1, 1]
        r_e = self.e.repeat(repeat_shape) #labeled
        dm = torch.Tensor([i for i in range(self.opt.domain_num)]).view(-1, 1).repeat(1, b).view(-1, 1).cuda() 
        
        r_et = self.et

        self.d_fake = self.netD(r_e, dm).view(self.opt.domain_num, -1)
        fake_weight = self.labeled_alpha[:, self.dm.long()]
        self.d_real = self.netD(r_et, self.dmt)
        real_weight = self.tr_im[self.dmt.long()].view(-1, 1)# balance sample numbers in N domains
        
        fake_one = 1.0*(self.d_fake>0.5) +0.5*(self.d_fake==0.5)
        fake_acc = fake_one*fake_weight
        fake_acc = fake_acc.sum()/fake_weight.sum()
        
        real_zero = 1.0*(self.d_real<0.5) +0.5*(self.d_real==0.5)
        real_acc = real_zero*real_weight
        real_acc = real_acc.sum()/real_weight.sum()

        self.loss_A_gan = fake_acc+real_acc-1.0
         
        sum_alpha = self.sum_alpha/(self.opt.domain_num*self.beta)
        big_alpha = self.lambda_3rd_term * self.big_alpha/(self.opt.domain_num*self.beta.view(-1, 1))
        all_alpha = torch.cat((sum_alpha.view(1, -1), big_alpha), dim=0)
        C_weight = all_alpha[:, self.dm.long()].t()
        pred_label = torch.argmax(self.out.view(-1, self.opt.domain_num+1 ,self.opt.class_num).detach(), dim=2)
        cls_wrong = (pred_label!=self.y.view(-1, 1).repeat(1, self.opt.domain_num+1))
        self.loss_A_pred = cls_wrong*C_weight
        self.loss_A_pred = self.loss_A_pred.sum()/C_weight.sum()
        self.loss_A = self.lambda_A*self.loss_A_gan + 2*self.loss_A_pred
        
        self.loss_A.backward(retain_graph=True) # self.loss_D
        self.set_alpha()

    def backward_prob_A(self):
        """
        fake: labeled domain, groundtruth is one
        real: orginal domain, groundtruth is zero
        """
        self.set_alpha()
        b = self.e.size(0)
        repeat_shape = [self.domain_num]+ [1 for i in range(len(self.e.shape)-1)] # if len(self.e.shape)=2, [domain_num, 1]
        # if len(self.e.shape)=4, [domain_num, 1, 1, 1]
        r_e = self.e.repeat(repeat_shape) #labeled
        dm = torch.Tensor([i for i in range(self.opt.domain_num)]).view(-1, 1).repeat(1, b).view(-1, 1).cuda() 
        
        r_et = self.et

        self.d_fake = self.netD(r_e, dm).view(self.opt.domain_num, -1)
        fake_weight = self.labeled_alpha[:, self.dm.long()]
        self.d_real = self.netD(r_et, self.dmt)
        real_weight = self.tr_im[self.dmt.long()].view(-1, 1)# balance sample numbers in N domains
        
        fake_one = self.d_fake
        fake_acc = fake_one*fake_weight
        fake_acc = fake_acc.sum()/fake_weight.sum()
        
        real_zero = 1.0-self.d_real
        real_acc = real_zero*real_weight
        real_acc = real_acc.sum()/real_weight.sum()

        self.loss_A_gan = fake_acc+real_acc-1.0
         
        sum_alpha = self.sum_alpha/(self.opt.domain_num*self.beta)
        big_alpha = self.lambda_3rd_term * self.big_alpha/(self.opt.domain_num*self.beta.view(-1, 1))
        all_alpha = torch.cat((sum_alpha.view(1, -1), big_alpha), dim=0)
        C_weight = all_alpha[:, self.dm.long()].t()

        pred_prob = torch.softmax(self.out.view(-1, self.opt.domain_num+1, self.opt.class_num).detach(), dim=2)
        # correct_index = self.y.view(-1, 1, 1).repeat(1, self.opt.domain_num+1, 1).long()
        # print("correct_index.shape", correct_index.shape)
        correct_prob = pred_prob[range(pred_prob.shape[0]), :, self.y.long()].view(-1, self.opt.domain_num+1)
        # print("correct_prob.shape", correct_prob.shape)
        wrong_prob = 1.0 - correct_prob

        self.loss_A_pred = wrong_prob*C_weight
        self.loss_A_pred = self.loss_A_pred.sum()/C_weight.sum()
        self.loss_A = self.lambda_A*self.loss_A_gan + 2*self.loss_A_pred
        
        self.loss_A.backward(retain_graph=True) # self.loss_D
        self.set_alpha()

    def backward_A(self):
        self.set_alpha()
        b = self.e.size(0)
        repeat_shape = [self.domain_num]+ [1 for i in range(len(self.e.shape)-1)] # if len(self.e.shape)=2, [domain_num, 1]
        # if len(self.e.shape)=4, [domain_num, 1, 1, 1]
        r_e = self.e.repeat(repeat_shape) #labeled
        dm = torch.Tensor([i for i in range(self.opt.domain_num)]).view(-1, 1).repeat(1, b).view(-1, 1).cuda() 
        
        r_et = self.et

        self.d_fake = self.netD(r_e, dm).view(self.opt.domain_num, -1)
        fake_weight = self.labeled_alpha[:, self.dm.long()]
        self.d_real = self.netD(r_et, self.dmt)
        real_weight = self.tr_im[self.dmt.long()].view(-1, 1)# balance sample numbers in N domains
        
        fake_label = torch.zeros_like(self.d_fake).cuda()
        fake_loss = self.adv_loss(self.d_fake, fake_label)*fake_weight
        fake_loss = fake_loss.sum()/fake_weight.sum()
        
        real_label = torch.ones_like(self.d_real).cuda()
        real_loss = self.adv_loss(self.d_real, real_label)*real_weight
        real_loss = real_loss.sum()/real_weight.sum()

        self.loss_A_gan = (fake_loss + real_loss)/2.0
         
        sum_alpha = self.sum_alpha/(self.opt.domain_num*self.beta)
        big_alpha = self.lambda_3rd_term * self.big_alpha/(self.opt.domain_num*self.beta.view(-1, 1))
        all_alpha = torch.cat((sum_alpha.view(1, -1), big_alpha), dim=0)
        C_weight = all_alpha[:, self.dm.long()].t()
        # print(self.out.view(-1, self.opt.class_num).shape, self.y.view(-1, 1).repeat(1, self.opt.domain_num+1).view(-1).shape)
        self.loss_A_pred = self.ce_loss(self.out.view(-1, self.opt.class_num), self.y.view(-1, 1).repeat(1, self.opt.domain_num+1).view(-1)).view(-1, self.opt.domain_num+1)*C_weight
        self.loss_A_pred = self.loss_A_pred.sum()/C_weight.sum()
        self.loss_A = self.loss_A_gan * self.lambda_A + self.loss_A_pred
        
        self.loss_A.backward(retain_graph=True) # self.loss_D
        self.set_alpha()
    
    def backward_G_alphaDA(self):
        self.set_alpha()
        b = self.e.size(0)
        repeat_shape = [self.domain_num]+ [1 for i in range(len(self.e.shape)-1)] # if len(self.e.shape)=2, [domain_num, 1]
        # if len(self.e.shape)=4, [domain_num, 1, 1, 1]
        r_e = self.e.repeat(repeat_shape)
        dm = torch.Tensor([i for i in range(self.opt.domain_num)]).view(-1, 1).repeat(1, b).view(-1, 1).cuda() 
        
        r_et = self.et

        self.d_fake = self.netD(r_e, dm).view(self.opt.domain_num, -1)
        fake_weight = self.labeled_alpha[:, self.dm.long()]
        self.d_real = self.netD(r_et, self.dmt)
        real_weight = self.tr_im[self.dmt.long()].view(-1, 1)# balance sample numbers in N domains

        fake_label = torch.zeros_like(self.d_fake).cuda()
        fake_loss = self.adv_loss(self.d_fake, fake_label)*fake_weight
        fake_loss = fake_loss.sum()/fake_weight.sum()
        
        real_label = torch.ones_like(self.d_real).cuda()
        real_loss = self.adv_loss(self.d_real, real_label)*real_weight
        real_loss = real_loss.sum()/real_weight.sum()

        self.loss_E_gan = (fake_loss + real_loss)/2.0
         
        sum_alpha = self.sum_alpha/(self.opt.domain_num*self.beta)
        big_alpha = self.lambda_3rd_term * self.big_alpha/(self.opt.domain_num*self.beta.view(-1, 1))
        all_alpha = torch.cat((sum_alpha.view(1, -1), big_alpha), dim=0)
        C_weight = all_alpha[:, self.dm.long()].t()
        self.loss_E_pred = self.ce_loss(self.out.view(-1, self.opt.class_num), self.y.view(-1, 1).repeat(1, self.opt.domain_num+1).view(-1)).view(-1, self.opt.domain_num+1)*C_weight
        self.loss_E_pred = self.loss_E_pred.sum()/C_weight.sum()
        # self.loss_E_entropy = EntropyLoss(self.softmax(self.outt[:, :self.opt.class_num]))
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred #+ self.lambda_e * self.loss_E_entropy
        
        self.loss_E.backward(retain_graph=True) # self.loss_D
        self.set_alpha()
    
    def backward_D(self): #_alphaDA
        
        """
        align unlabeled data in per domain and all labeled data
    
        fake: labeled domain, groundtruth is one
        real: orginal domain, groundtruth is zero
        
        """

        self.set_alpha()

        repeat_shape = [self.domain_num]+ [1 for i in range(len(self.e.shape)-1)] # if len(self.e.shape)=2, [domain_num, 1]
        # if len(self.e.shape)=4, [domain_num, 1, 1, 1]
        r_e = self.e.repeat(repeat_shape)
        
        b = self.e.size(0)
        dm = torch.Tensor([i for i in range(self.opt.domain_num)]).view(-1, 1).repeat(1, b).view(-1, 1).cuda() 
        
        r_et = self.et

        self.d_fake = self.netD(r_e, dm).view(self.opt.domain_num, -1)
        fake_weight = self.labeled_alpha[:, self.dm.long()]
        self.d_real = self.netD(r_et, self.dmt)
        real_weight = self.tr_im[self.dmt.long()].view(-1, 1)# balance sample numbers in N domains

        fake_label = torch.ones_like(self.d_fake).cuda()
        fake_loss = self.adv_loss(self.d_fake, fake_label)*fake_weight
        fake_loss = fake_loss.sum()/fake_weight.sum()
        
        real_label = torch.zeros_like(self.d_real).cuda()
        real_loss = self.adv_loss(self.d_real, real_label)*real_weight
        real_loss = real_loss.sum()/real_weight.sum()

        self.loss_D = (fake_loss + real_loss)/2.0
        # self.loss_E_gan = self.loss_D
        self.loss_D.backward(retain_graph=True)
        self.set_alpha()
      
    def backward_G(self):
        self.set_alpha()

        b = self.e.size(0)
        repeat_shape = [self.domain_num]+ [1 for i in range(len(self.e.shape)-1)] # if len(self.e.shape)=2, [domain_num, 1]
        # if len(self.e.shape)=4, [domain_num, 1, 1, 1]
        r_e = self.e.repeat(repeat_shape)
        dm = torch.Tensor([i for i in range(self.opt.domain_num)]).view(-1, 1).repeat(1, b).view(-1, 1).cuda() 
        # 6*1->6*b->6b*1
        r_et = self.et

        self.d_fake = self.netD(r_e, dm).view(self.opt.domain_num, -1)
        fake_weight = self.labeled_alpha[:, self.dm.long()]
        self.d_real = self.netD(r_et, self.dmt)
        real_weight = self.tr_im[self.dmt.long()].view(-1, 1)# balance sample numbers in N domains

        fake_label = torch.zeros_like(self.d_fake).cuda()
        fake_loss = self.adv_loss(self.d_fake, fake_label)*fake_weight
        fake_loss = fake_loss.sum()/fake_weight.sum()
        
        real_label = torch.ones_like(self.d_real).cuda()
        real_loss = self.adv_loss(self.d_real, real_label)*real_weight
        real_loss = real_loss.sum()/real_weight.sum()
        
        self.loss_E_gan = (fake_loss + real_loss)/2.0

        # self.loss_D = 
        # self.loss_E_gan = self.loss_D

        all_alpha = 1.0/(self.opt.domain_num*self.beta)
        C_weight = all_alpha[self.dm.long()]
        self.loss_E_pred = self.ce_loss(self.out[:, :self.opt.class_num], self.y)*C_weight
        self.loss_E_pred = self.loss_E_pred.sum()/C_weight.sum()
        # self.loss_E_entropy = EntropyLoss(self.softmax(self.outt[:, :self.opt.class_num]))
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred #+ self.lambda_e * self.loss_E_entropy
        
        self.loss_E.backward(retain_graph=True) # self.loss_D
        self.set_alpha()

    def optimize_parameters_alphaDA(self):
        self.forward_train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad() #_alphaDA
        self.backward_D()
        self.optimizer_D.step()
        # # update alpha
        self.set_requires_grad(self.netD, False)
        self.optimizer_A.zero_grad()
        self.backward_A()
        self.optimizer_A.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G_alphaDA()
        self.optimizer_G.step()
    
    def optimize_parameters_acc_alphaDA(self):
        self.forward_train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad() #_alphaDA
        self.backward_D()
        self.optimizer_D.step()
        # # update alpha
        self.set_requires_grad(self.netD, False)
        self.optimizer_A.zero_grad()
        self.backward_acc_A()
        self.optimizer_A.step()
        # # update D % add it in 2022-05-24
        # self.set_requires_grad(self.netD, True)
        # self.optimizer_D.zero_grad() #_alphaDA
        # self.backward_D()
        # self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G_alphaDA()
        self.optimizer_G.step()
    
    def optimize_parameters_prob_alphaDA(self):
        self.forward_train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad() #_alphaDA
        self.backward_D()
        self.optimizer_D.step()
        # # update alpha
        self.set_requires_grad(self.netD, False)
        self.optimizer_A.zero_grad()
        self.backward_prob_A()
        self.optimizer_A.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G_alphaDA()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward_train()
        # update D
        # for i in range(2):
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    

    def test(self, dataloader):
        # print('test!')
        self.eval()
        # print('self.eval()')
        self.acc_reset()
        # print('self.acc_reset()')
        # print(dataloader)
        # print(len(dataloader))

        
        with torch.no_grad():
            len_dataset = len(dataloader)
            # print(len_dataset)
            # t1 = time.time()
            iter_loader = iter(dataloader)
            # t2 = time.time()
            # print(f'time:{t2-t1}')
            for i in range(len_dataset):
                # print('testing')
                x, y, dm, _, _ = iter_loader.next()
            # for x, y, dm, _, idxs in dataloader:
                
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                self.set_input(input=(x, y, dm))
                self.forward()
                self.acc_update()
            # t3 = time.time()
            # print(f'time:{t3-t2}')
        self.acc_calc()
        self.acc_test = self.acc
        self.best_acc_test = max(self.best_acc_test, self.acc_test)
        if self.best_acc_test == self.acc_test:
            self.best_epoch = self.epoch
            self.save()
            self.save_para()
            self.logger_train.info('big_alpha {}'.format(self.big_alpha))

        self.acc_msg = '[Round][{}][Test][{}] best test[{}] {:.5f} test {:.5f}'.format(
            self.rd, self.epoch, self.best_epoch, self.best_acc_test, self.acc_test)
        self.loss_msg = ''
        self.print_log()
        # print('test finished')
    def val_poor_ratio(self, x, y, dm, prior=0.5):
        # print('test!')
        self.eval()

        
        with torch.no_grad():
            x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
            self.set_input(input=(x, y, dm))
            self.forward()
            
        Y = to_np(self.y)
        P = to_np(self.pred)
        D = to_np(self.dm)
        D = (D).astype(np.int32)
        # print('shape',(Y == P).shape)
        hit = (Y == P).astype(np.float32)
        hit_domain, cnt_domain = np.zeros(self.opt.domain_num), np.zeros(self.opt.domain_num)
        for i in range(self.opt.domain_num):
            hit_domain[i] += hit[D == i].sum()
            cnt_domain[i] += (D == i).sum()
    
        error_domain =1.0 - hit_domain / cnt_domain
        ratio = prior*(1.0/self.opt.domain_num)+(1-prior)*(error_domain/error_domain.sum())
        return ratio, error_domain

                
    def get_mu(self):
        return self.load_npy(str(self.rd))['mu']
