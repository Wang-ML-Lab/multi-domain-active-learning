import numpy as np
from .strategy import Strategy
import pdb
import numpy as np
from .strategy import Strategy
from copy import copy as copy
from copy import deepcopy as deepcopy
import pdb
from scipy import stats
import torch
from sklearn.metrics import pairwise_distances


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

class ganSampling(Strategy):
	def __init__(self, train_set, test_set, handler, net, opts):
		super(ganSampling, self).__init__(train_set, test_set, handler, net, opts)
    
	def query(self, alpha, rd):
		"""
		alpha: a list contain weights of every domain
		rd: round
		"""

		lb_num = self.n_init + self.n_query*rd # total labelling number
		new_lb_idx = []
		idxs_lb = self.idxs_lb
		existed_label = np.array([sum(self.idxs_lb[self.dm==i]) for i in range(self.dm_num)])
		###
		existed_alpha = np.array([existed_label[i]/lb_num*1.0 for i in range(alpha.shape[0])])
		# in case that the number of existed labels is bigger than the number of data that should be labeled in this round
		def cal_new_alpha(current_alpha):
			new = np.array([max(current_alpha[i], existed_alpha[i]) for i in range(alpha.shape[0])])
			new[new -current_alpha ==0] = (1.0- new[new -current_alpha > 0].sum()) * new[new -current_alpha ==0] / new[new -current_alpha ==0].sum()
			if min(new-existed_alpha)>=0.0:
				print('pos',new)
				return new
			else:
				print('neg',new)
				return cal_new_alpha(new)
		alpha_new = cal_new_alpha(alpha)
		# print(alpha_new)
		label_per_domain = np.array([int(round(lb_num*alpha_new[i], 12)) for i in range(alpha.shape[0])])
		###
		rest_label = lb_num-label_per_domain.sum() # rest budget
		
		if rest_label !=0:
			rest_point_num = np.array([(lb_num*alpha_new[i])%1 for i in range(self.dm_num)])
			y = np.argsort(-rest_point_num) # for big to small
			label_per_domain[y[:rest_label]] +=1
		per_domain_num = label_per_domain-existed_label # the number of data should be labeled in every domain
		# print("alpha, alpha_new, per_domain_num, label_per_domain, existed_label", alpha, alpha_new, per_domain_num, label_per_domain, existed_label)
		for i in range(self.dm_num):

			if per_domain_num[i]==0:
				new_add = []
			else:
				idxs_unlabeled = np.where((self.dm==i)&(idxs_lb==0))[0] # i-th domain without labels
				# labeled index of current round, current domain
				#——————————————————————————————————————————————————————————————————————————————
				# 1st to 7th 9th
				U = self.predict_gan_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.dm[idxs_unlabeled], self.idxs_lb[idxs_unlabeled])
				
				

				cls_gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.dm[idxs_unlabeled], self.idxs_lb[idxs_unlabeled])
				gan_gradEmbedding = self.opt.gan_uncertainty*self.get_dis_grad_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.dm[idxs_unlabeled], self.idxs_lb[idxs_unlabeled])
				gradEmbedding = torch.cat((cls_gradEmbedding, gan_gradEmbedding), dim=1).numpy()
				chosen = init_centers(gradEmbedding, per_domain_num[i])
				new_add = idxs_unlabeled[chosen]

				

				
			#——————————————————————————————————————————————————————————————————————————————
			new_lb_idx.append(new_add)
			print(i, per_domain_num[i], len(new_add))
			print(new_add)

		new_lb_idx = [new_lb_idx[i][j] for i in range(len(new_lb_idx)) 
		for j in range(len(new_lb_idx[i]))]

		return new_lb_idx

