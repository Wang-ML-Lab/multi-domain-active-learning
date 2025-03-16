from asyncio.log import logger
import numpy as np
import sys
import time
import gzip
# import openml
import os
import argparse
from dataset import get_dataset, get_handler
from model_alpha import get_net
import query_strategies
import resnet
import random
from networks import *
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import pdb
from scipy.stats import zscore
import logging
from query_strategies import RandomSampling, BadgeSampling, ISAL_Sampling, FMSampling, \
                                CMSampling, BaselineSampling, LeastConfidence, MarginSampling, \
                                EntropySampling, CoreSet, ActiveLearningByLearning, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, \
                                    RandomSampling1d, MarginSampling1d, BadgeSampling1d, \
                                        CMSampling1d, FMSampling1d, ganSampling
import multiprocessing
import pre_process
# code based on https://github.com/ej0cl6/deep-active-learning"
os.environ['TORCH_HOME']='/home/dycpu1/gyh/pycharm/badge1/pretrained_models/'
def one_experiment(opts):
    """ 
    gpu_id
    1 dataset-args: data
        (setting of datasets should be in this function, such as batch_size, max_epoch)
        1 query strategy-args: query
            1 model---(10 rounds)-args:model, lr, etc.
                1 seed-args: seed
    """
    #pre-1 set model save dirs
    opts.out_fold = './results-mnist/' + opts.data + '/' + opts.query +'/' + opts.self_define_name  # decided by dataset/query
    opts.save_name = opts.out_fold + '/' + opts.model # decided by dataset/query/model
    os.system('mkdir -p ' + opts.out_fold) # create dir
    print('Train result will be saved in ', opts.out_fold)
    #pre-2 log training precess and final results
    log_train = opts.save_name+'_'+ str(opts.seed)+ '_train.log' #decided by dataset/query/model/seed
    log_final = opts.save_name+'_'+ str(opts.seed)+ '_final.log'
    def setup_logger(name, log_file, level=logging.INFO):
        """To setup as many loggers as you want"""
        # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        # formatter = logging.Formatter('%(message)s')
        handler = logging.FileHandler(log_file)        
        # handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

    opts.logger_train = setup_logger('train_logger', log_train)
    opts.logger_final = setup_logger('final_logger', log_final)

    # training setting————————————————————————————————————————————
    #1 set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_id)

    #2 set dataset
    #2-1 dataset defaults
    args_pool = {'MNIST':
                    {'n_epoch': 500, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                    'loader_tr_args':{'batch_size': 100, 'num_workers': 1},
                    'loader_label_args':{'batch_size': 100, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 100, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                    }
    args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']


    opts.args = args_pool[opts.data]
    if not os.path.exists(opts.path):
        os.makedirs(opts.path)
    
    #2-2 process datasets into array form
    if opts.data == 'MNIST':
        
        def angle_list(domain_num=8, per_angle=15):
            # if rotation angles are uniform, use this function
            al = []
            for i in range(domain_num):
                al.append(i* per_angle)
                al.append((i+1)* per_angle)
            return al

        opts.rotate_angle = angle_list(opts.domain_num, opts.rotate_angle)
        
        tr, te = get_dataset(opts.data, opts.path, opts.rotate_angle)

        X_tr, Y_tr, dm_tr, dm_sample_num_tr, dm_sample_index_tr, angle_tr = tr
        # tr: train_set, X_tr: images, Y_tr: labels, dm_tr: domain index (e.g., 0-5 if there are six domains), 
        # dm_sample_num_tr: the number of images in each domain, dm_sample_index_tr: image index of each domain, angle_tr: image's rotation angle
        X_te, Y_te, dm_te, dm_sample_num_te, dm_sample_index_te, angle_te = te
       
    opts.beta = dm_sample_num_tr/dm_sample_num_tr.sum()

    n_total = len(Y_tr) # total data number
    opts.dim = np.shape(X_tr)[1:]
    idxs_lb = np.zeros(n_total, dtype=bool) #  labeled train image is True
    idxs_lb_te =np.ones(n_total, dtype=bool)
    train_set = X_tr, Y_tr, dm_tr, idxs_lb 
    test_set = X_te, Y_te, dm_te, idxs_lb_te


    handler = get_handler(opts.data)

    #3 set model
    if opts.data == 'MNIST':
        opts.lr = 0.0002
        opts.class_num=10
        opts.beta1=0.9
        opts.weight_decay = 0.0005
        opts.nh = 256
        
    else: 
        raise ValueError
    net = get_net()(opts)

    #4 set query strategy and collect all of things into a whole function of training
    # set up the specified sampler
    if opts.query == 'random': # random sampling
        strategy = RandomSampling(train_set, test_set, handler, net, opts)
    elif opts.query == 'grads': # Gradient with Discriminator Score (GraDS) sampling
        strategy = ganSampling(train_set, test_set, handler, net, opts)
    elif opts.query == 'entropy': # entropy-based sampling
        strategy = EntropySampling(train_set, test_set, handler, net, opts)
    elif opts.query == 'margin': # margin-based sampling
        strategy = MarginSampling(train_set, test_set, handler, net, opts)
    elif opts.query == 'badge': # batch active learning by diverse gradient embeddings
        strategy = BadgeSampling(train_set, test_set, handler, net, opts)
    elif opts.query == 'cm': # cluster-margin sampling
        strategy = CMSampling(train_set, test_set, handler, net, opts)
    elif opts.query == 'fm': # free energy+margin sampling
        strategy = FMSampling(train_set, test_set, handler, net, opts)
    elif opts.query == 'random1d': # random sampling, treat all domains as a single domain
        strategy = RandomSampling1d(train_set, test_set, handler, net, opts)
    elif opts.query == 'margin1d': # margin-based sampling, treat all domains as a single domain
        strategy = MarginSampling1d(train_set, test_set, handler, net, opts)
    elif opts.query == 'badge1d': # batch active learning, treat all domains as a single domain
        strategy = BadgeSampling1d(train_set, test_set, handler, net, opts)
    elif opts.query == 'cm1d': # cluster-margin sampling, treat all domains as a single domain
        strategy = CMSampling1d(train_set, test_set, handler, net, opts)
    elif opts.query == 'fm1d': # free energy+margin sampling, treat all domains as a single domain
        strategy = FMSampling1d(train_set, test_set, handler, net, opts)
    else: 
        print('choose a valid acquisition function', flush=True)
        raise ValueError

    #5 print hyperparameters
    for k in list(vars(opts).keys()):
        opts.logger_final.info('%s: %s' % (k, vars(opts)[k])) # save hyperpara
    # 6 training process
    # print data dir and strategy name
    opts.logger_train.info(opts.data)
    opts.logger_train.info(type(strategy).__name__)
    # query parameters
    opts.logger_train.info('number of labeled pool: {}'.format(opts.nStart))
    opts.logger_train.info('number of unlabeled pool: {}'.format(n_total - opts.nStart))
    opts.logger_train.info('number of testing pool: {}'.format(len(Y_te)))
    NUM_ROUND = int((opts.nEnd - opts.nStart)/ opts.nQuery)
    ####Training Rounds####
    # Round 0
    # generate initial labeled pool
    # query
    def set_beta(idx_lb):
        # caculate budgets
        l =[]
        for i in range(opts.domain_num):
            l.append(sum(idx_lb[dm_tr==i]))
        return (torch.Tensor(l)/torch.Tensor(l).sum()).cuda()

    opts.logger_train.info("Round {} train ratio:{}".format(0, opts.beta*opts.domain_num))
    
    ### function to randomly label data in the round 0

    def random_query0(alpha_new):
        """
        alpha: a list contain learned similarity weights of every domain
        rd: round
        """
        lb_num = opts.nStart
        new_lb_idx = []
        existed_label = np.array([sum(idxs_lb[dm_tr==i]) for i in range(opts.domain_num)])
        label_per_domain = np.array([int(lb_num*alpha_new[i]) for i in range(alpha_new.shape[0])])
        rest_label = lb_num-label_per_domain.sum() # rest budget

        if rest_label !=0:
            rest_point_num = np.array([(lb_num*alpha_new[i])%1 for i in range(opts.domain_num)])
            y = np.argsort(-rest_point_num) # for big to small
            label_per_domain[y[:rest_label]] +=1
        per_domain_num = label_per_domain-existed_label # the number of data should be labeled in every domain
        # print("alpha, alpha_new, per_domain_num, label_per_domain, existed_label", alpha, alpha_new, per_domain_num, label_per_domain, existed_label)
        for i in range(opts.domain_num):
            i_idxs_no_label = np.where((dm_tr==i)&(idxs_lb==0))[0]
            # labelling number of current round, current domain
            new_add = i_idxs_no_label[np.random.permutation(len(i_idxs_no_label))][:per_domain_num[i]]
            new_lb_idx.append(new_add)
            print(i, per_domain_num[i], len(new_add))
            print(new_add)

        new_lb_idx = [new_lb_idx[i][j] for i in range(len(new_lb_idx)) 
        for j in range(len(new_lb_idx[i]))]
        
        return new_lb_idx


    q_idxs = random_query0(np.ones(opts.domain_num)/opts.domain_num) # random sample images to label in each domain in the round 0; q_idxs: indices of labeled images
    idxs_lb[q_idxs] = True
    # update
    net.beta = set_beta(idxs_lb) # beta: label budget ratio
    opts.logger_train.info('beta:{}'.format(net.beta*opts.domain_num))
    strategy.update(idxs_lb)
    # train round 0
    strategy.train(0)

    # Round 1 to NUM_ROUND
    for rd in range(1, NUM_ROUND+1):
        
        opts.logger_train.info("Round {} mu: {}".format(rd, net.get_mu()*opts.domain_num))
        # query
        q_idxs = strategy.query(net.get_mu(), rd) # mu (alpha) is ratio to distribute labels to domains
        idxs_lb[q_idxs] = True
        net.beta = set_beta(idxs_lb)
        opts.logger_train.info('beta: {}'.format(net.beta*opts.domain_num))

        # update
        strategy.update(idxs_lb)
        strategy.train(rd)
        
        if sum(~strategy.idxs_lb) < opts.nQuery: 
            sys.exit('too few remaining points to query')
        # print(sum(strategy.idxs_lb), sum(~strategy.idxs_lb)) # 600 59400 ~:get inversed value
    opts.logger_train.info("Round {} mu:{}".format(100, net.get_mu()*opts.domain_num)) # after training, return final alpha
    return

if __name__ == "__main__":

    def pretrain(para_list):

        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu_id', type=int, default=para_list[0], help="device id to run")# gpu
        parser.add_argument('--data', help='datasets', type=str, default=para_list[1]) # dataset
        parser.add_argument('--domain_num', help='the number of domains', type=int, default=6) # the number of domains
        parser.add_argument('--rotate_angle', help='the rotation angle of every domain', type=int, default=180.0/6.0) # rotation angle range of each domain
        parser.add_argument('--query', help='acquisition algorithm', type=str, default=para_list[2]) # query strategy
        parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=para_list[3]) # label budget when round>0
        parser.add_argument('--nStart', help='number of points to start', type=int, default=para_list[4]) # label budget when round=0
        parser.add_argument('--nEnd', help = 'total number of labels', type=int, default=para_list[5]) # total budget
        parser.add_argument('--model', help='pure, alpha, DA, alDA', type=str, default=para_list[6]) 
        """
        pure: Traditional methods using only instance-level strategies.
        DA: Ablation study of our proposed CAL with only feature alignment.
        alpha: Ablation study of our proposed CAL with only similarity estimation.
        alDA: Our proposed CAL.
        """
        parser.add_argument('--lr', help='learning rate', type=float, default=para_list[7]) # model
        parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256) 
        parser.add_argument('--nz', help='number of embedding dims (conv)', type=int, default=100)  
        parser.add_argument('--dropout', help='dropout rate', type=float, default=0.0) 
        parser.add_argument('--seed', help='random seed', type=int, default=para_list[8])
        parser.add_argument('--path', help='data path', type=str, default='data')
        opts = parser.parse_args()
        
        # set seed
        def set_seed(seed):
            """Set all random seeds."""
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False # false: not optimize speed, suit to dynamic net
        set_seed(opts.seed)
        
        if 'DA' in opts.model:
            opts.lambda_gan = para_list[9] # hyperparameter for domain gap term in feature alignment
        else:
            opts.lambda_gan = 0
        
        if 'alpha' in opts.model:
            opts.alpha_update='acc'  
            if 'DA' in opts.model: # alphaDA
                opts.lambda_A=1 # hyperparameter for domain gap term in similiarity estimation for alphaDA
            else:
                opts.lambda_A = para_list[9] # alpha
              
        else:
            opts.lambda_A = 0
        
        if len(para_list) > 10:
            opts.domain_num = para_list[10]
        
        domain_str = 'd'+str(opts.domain_num)

        opts.diff_cls = False
        if opts.diff_cls:
            cls_str = 'diff'
        else:
            cls_str = 'same'
        opts.alignment_ablation = False # do not have ablation in this part
        opts.third_term_ablation = False

        if opts.model=='alphaDA':
            opts.self_define_name = domain_str + '_' + cls_str + '_' + str(opts.lambda_gan)+ '_' + str(opts.lambda_A) 
        else:
            opts.self_define_name = domain_str + '_' + cls_str + '_' + str(max(opts.lambda_gan, opts.lambda_A))
        if query_method=='fm' or 'fm1d':
            opts.eng_w = 0.5
        if query_method=='grads':
            opts.gan_uncertainty = 0.1
        one_experiment(opts)
        return


lr = 0.0002
seed_1 = 101
seed_2 = 20
seed_3 = 11

num_init_round = 150
num_per_round = 150
num_round = 5
num_end = num_init_round + num_round*num_per_round

 

data_name = 'MNIST'

"""
CAL contains two levels of query strategy:

(1) Instance-level strategy: Selects samples from unlabeled data, including:
    - Random sampling ('random')
    - Entropy-based sampling ('entropy')
    - GraDS sampling ('grads') – our proposed instance-level strategy (see Sec. 3.5 of our paper)
    - Other strategies (refer to lines 134–157 for details)

(2) Domain-level strategy: Determines how many samples should be selected from each domain.

The following demonstrates how to set hyperparameters:

Example:

query_method = 'grads'  # Choice of instance-level strategy

# Decoder: CAL contains two key elements: feature alignment and similarity estimation
# The following sets the lambda for CAL in Equation 5.
da = 2        # Sets lambda_d to 2 for the ablation study of CAL with only feature alignment
alpha = 1     # Sets lambda_d to 1 for the ablation study of CAL with only similarity estimation
alpha_da = 1  # Sets lambda_d to 1 in CAL

# To run traditional methods (using only instance-level strategies),
# set the "model" argument of the pretrain function (line 251) to 'pure'.
# This can be done in para_list (see below).
# For an explanation of hyperparameters in para_list, refer to the function pretrain (line 251).
"""

query_method = 'grads' 
da =  2 #1
alpha = 1#0.3 #1
alpha_da = 1 #1

# query_method = 'random' 
# da = 2
# alpha = 1
# alpha_da = 1

# query_method = 'margin' 
# alpha = 1
# alpha_da = 1

# query_method = 'badge' 
# da = 2
# alpha = 1
# alpha_da = 0.5

# query_method = 'cm'
# da = 2
# alpha = 1
# alpha_da = 1

# query_method = 'fm' 
# da = 2 #2
# alpha = 1
# alpha_da = 1 #1



gpu_pure = 0
gpu_da = 1
gpu_al = 2
gpu_ad = 3
para_list1 = [gpu_pure, data_name, query_method, num_per_round, num_init_round, num_end, 'pure', lr, seed_1]
# para_list2 = [gpu_pure, data_name, query_method, num_per_round, num_init_round, num_end, 'pure', lr, seed_2]
# para_list3 = [gpu_pure, data_name, query_method, num_per_round, num_init_round, num_end, 'pure', lr, seed_3]

para_list4 = [gpu_da, data_name, query_method, num_per_round, num_init_round, num_end, 'DA', lr, seed_1, da]
# para_list5 = [gpu_da, data_name, query_method, num_per_round, num_init_round, num_end, 'DA', lr, seed_2, da]
# para_list6 = [gpu_da, data_name, query_method, num_per_round, num_init_round, num_end, 'DA', lr, seed_3, da]

para_list7 = [gpu_al, data_name, query_method, num_per_round, num_init_round, num_end, 'alpha', lr, seed_1, alpha]
# para_list8 = [gpu_al, data_name, query_method, num_per_round, num_init_round, num_end, 'alpha', lr, seed_2, alpha]
# para_list9 = [gpu_al, data_name, query_method, num_per_round, num_init_round, num_end, 'alpha', lr, seed_3, alpha]

para_list10 = [gpu_ad, data_name, query_method, num_per_round, num_init_round, num_end, 'alphaDA', lr, seed_1, alpha_da] 
# para_list11 = [gpu_ad, data_name, query_method, num_per_round, num_init_round, num_end, 'alphaDA', lr, seed_2, alpha_da]
# para_list12 = [gpu_ad, data_name, query_method, num_per_round, num_init_round, num_end, 'alphaDA', lr, seed_3, alpha_da]


p1 = multiprocessing.Process(target=pretrain, args=(para_list1,))
# p2 = multiprocessing.Process(target=pretrain, args=(para_list2,))
# p3 = multiprocessing.Process(target=pretrain, args=(para_list3,))
p4 = multiprocessing.Process(target=pretrain, args=(para_list4,))
# p5 = multiprocessing.Process(target=pretrain, args=(para_list5,))
# p6 = multiprocessing.Process(target=pretrain, args=(para_list6,))
p7 = multiprocessing.Process(target=pretrain, args=(para_list7,))
# p8 = multiprocessing.Process(target=pretrain, args=(para_list8,))
# p9 = multiprocessing.Process(target=pretrain, args=(para_list9,))
p10 = multiprocessing.Process(target=pretrain, args=(para_list10,))
# p11 = multiprocessing.Process(target=pretrain, args=(para_list11,))
# p12 = multiprocessing.Process(target=pretrain, args=(para_list12,))

# ps = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
ps = [p1, p4, p7, p10]

for p in ps:
    p.start()
    # time.sleep(600)
for p in ps:
    p.join()

# python run.py > log/mnist_rd_same.log&
# python run.py > log/mnist_mg_same.log&
# python run.py > log/mnist_bd_same.log&
# python run.py > log/mnist_cm_same.log&
