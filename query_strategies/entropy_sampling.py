import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, train_set, test_set, handler, net, opts):
		super(EntropySampling, self).__init__(train_set, test_set, handler, net, opts)
    
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
			i_idxs_no_label = np.where((self.dm==i)&(idxs_lb==0))[0] # i-th domain without labels
			# labeled index of current round, current domain
			#——————————————————————————————————————————————————————————————————————————————
			probs = self.predict_prob(self.X[i_idxs_no_label], self.Y[i_idxs_no_label], self.dm[i_idxs_no_label], self.idxs_lb[i_idxs_no_label])
			log_probs = torch.log(probs)
			U = (probs*log_probs).sum(1) # negative entropy
			new_add = i_idxs_no_label[U.sort()[1]][:per_domain_num[i]]
			#——————————————————————————————————————————————————————————————————————————————
			new_lb_idx.append(new_add)
			print(i, per_domain_num[i], len(new_add))
			print(new_add)

		new_lb_idx = [new_lb_idx[i][j] for i in range(len(new_lb_idx)) 
		for j in range(len(new_lb_idx[i]))]

		return new_lb_idx	