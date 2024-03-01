from cProfile import label
import numpy as np
from .strategy import Strategy
import pdb

class RandomSampling_poor(Strategy):
    def __init__(self, train_set, test_set, handler, net, opts):
        super(RandomSampling_poor, self).__init__(train_set, test_set, handler, net, opts)

    def query(self, alpha, rd):
        """
        alpha: a list contain weights of every domain
        rd: round
        """
        new_lb_idx = []
        idxs_lb = self.idxs_lb
        ###
        per_domain_num = np.array([int(round(self.n_query*alpha[i], 12)) for i in range(alpha.shape[0])])
        ###
        rest_label = self.n_query-per_domain_num.sum() # rest budget

        if rest_label !=0:
            rest_point_num = np.array([(self.n_query*alpha[i])%1 for i in range(self.dm_num)])
            y = np.argsort(-rest_point_num) # from big to small
            per_domain_num[y[:rest_label]] +=1 # the number of data should be labeled in every domain
        # print("alpha, alpha_new, per_domain_num, label_per_domain, existed_label", alpha, alpha_new, per_domain_num, label_per_domain, existed_label)
        for i in range(self.dm_num):
            if per_domain_num[i]==0:
                new_add = []
            else:
                i_idxs_no_label = np.where((self.dm==i)&(idxs_lb==0))[0] # i-th domain without labels
                # labeled index of current round, current domain
                #——————————————————————————————————————————————————————————————————————————————
                new_add = i_idxs_no_label[np.random.permutation(len(i_idxs_no_label))][:per_domain_num[i]] # new added index
            #——————————————————————————————————————————————————————————————————————————————
            new_lb_idx.append(new_add)
            print(new_add)
            print(i, per_domain_num[i], len(new_add))
            

        new_lb_idx = [new_lb_idx[i][j] for i in range(len(new_lb_idx)) 
        for j in range(len(new_lb_idx[i]))]
        
        return new_lb_idx
        # old_idxs_lb = self.idxs_lb
        # for i in range(self.dm_num):
        #     inds = np.where(self.idxs_lb[self.dm==i]==0)[0]
        #     per_domain_num = int(lb_num*alpha[i])-sum(self.idxs_lb[self.dm==i]==1)
        #     print(rd, i, per_domain_num, sum(self.idxs_lb[self.dm==i]==1))
        #     # labelling number of current round, current domain
        #     x= inds[np.random.permutation(len(inds))][:per_domain_num]
        #     print(x)
        #     self.idxs_lb[self.dm==i][x]=True
        #     print(self.idxs_lb[self.dm==i][x])
        #     print(sum(self.idxs_lb[self.dm==i]==1))
        # print(sum(self.idxs_lb^old_idxs_lb))
        # # new_lb_idx = [new_lb_idx[i][j] for i in range(len(new_lb_idx)) 
        # # for j in range(len(new_lb_idx[i]))]
        
        # return np.where((self.idxs_lb^old_idxs_lb)==1)[0]
