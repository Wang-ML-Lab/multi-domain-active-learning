import numpy as np
from .strategy import Strategy
import pdb

class MarginSampling1d(Strategy):
	def __init__(self, train_set, test_set, handler, net, opts):
		super(MarginSampling1d, self).__init__(train_set, test_set, handler, net, opts)
    
	def query(self, alpha, rd):
		"""
		alpha: a list contain weights of every domain
		rd: round
		"""

		# lb_num = self.n_init + self.n_query*rd # total labelling number
		idxs_lb = self.idxs_lb
		if rd ==0:
			new_add_num = self.n_init
		else:
			new_add_num  =self.n_query

		if new_add_num==0:
			new_add = []
		else:
			idxs_unlabeled = np.where(idxs_lb==0)[0] # data without labels
			# labeled index of current round for all domains
			#——————————————————————————————————————————————————————————————————————————————
			probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.dm[idxs_unlabeled], self.idxs_lb[idxs_unlabeled])
			probs_sorted, idxs = probs.sort(descending=True)
			U = probs_sorted[:, 0] - probs_sorted[:,1]
			new_add = idxs_unlabeled[U.sort()[1].numpy()][:new_add_num]
		#——————————————————————————————————————————————————————————————————————————————
		print(np.array([sum(self.idxs_lb[self.dm==i]) for i in range(self.dm_num)]), len(new_add))
		print(new_add)

		return new_add.tolist()
