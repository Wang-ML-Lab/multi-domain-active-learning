import numpy as np
from .strategy import Strategy
import pdb

class FMSampling1d(Strategy):
	def __init__(self, train_set, test_set, handler, net, opts):
		super(FMSampling1d, self).__init__(train_set, test_set, handler, net, opts)
    
		self.eng_w = opts.eng_w

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
			# weight1 and weight 2—————————————————————————————————————————————————————————————————————————
			eng_num = int(new_add_num*self.eng_w)
			mg_num = new_add_num - eng_num
			#free energy——————————————————————————————————————————————————————————————————————————————
			engs = self.predict_free_energy(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.dm[idxs_unlabeled], self.idxs_lb[idxs_unlabeled])
			engs_sorted, idxs = engs.sort(descending=True) # from highest to lowest free energy
			new_add = idxs_unlabeled[idxs.numpy()][:eng_num]
		#——————————————————————————————————————————————————————————————————————————————
			#margin——————————————————————————————————————————————————————————————————————————————
			if mg_num != 0:
				idxs_unlabeled = idxs_unlabeled[idxs.numpy()][eng_num:]
				j_engs = self.predict_joint_energy(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.dm[idxs_unlabeled], self.idxs_lb[idxs_unlabeled])
				j_engs_sorted, idxs = j_engs.sort(descending=True)
				U = j_engs_sorted[:, 0] - j_engs_sorted[:,1]
				
				new_add = np.concatenate((new_add, idxs_unlabeled[U.sort()[1].numpy()][:mg_num]))
			#——————————————————————————————————————————————————————————————————————————————
		print(np.array([sum(self.idxs_lb[self.dm==i]) for i in range(self.dm_num)]), len(new_add))
		print(new_add)

		return new_add.tolist()


