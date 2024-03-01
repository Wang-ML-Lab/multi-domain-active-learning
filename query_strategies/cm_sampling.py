import numpy as np
from .strategy import Strategy
import pdb
from sklearn.cluster import AgglomerativeClustering
import random

def HAC(X):
	"""0 round's clustering"""
	n_clu = 1
	dis = 3.0 # 5.7 #mnist: 5.85, 5.7 of: 0.1
	while n_clu < 5:
		dis += 0.1 #0.01
		clustering = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=dis).fit(X)
		# print(clustering.n_clusters_)
		# print(dis)
		n_clu = X.shape[0]/clustering.n_clusters_
		# print(n_clu)
	print(dis, n_clu)
	# clustering = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=dis).fit(X)
	# print(clustering.n_clusters_)
	total_labels = clustering.labels_ # init clusters

	return total_labels, clustering.n_clusters_, dis
def new_index(U, orginal_index, kt, total_labels):
	# U, idxs_unlabeled, kt, self.total_labels[i]
	km = 5*kt
	
	sort_index =np.argsort(U) # relative index
	sort_index = orginal_index[sort_index][:km] # absolute_index
	# print(sort_index)
	
	labels = total_labels[sort_index] #all N domains' total_labels
	# print("sort_index.shape", sort_index.shape)
	# print("len(labels)", len(labels))
	cluster_dict ={}
	for i in range(len(labels)):
		# print('labels[i]', labels[i], i)
		if labels[i] not in cluster_dict.keys():
			cluster_dict[labels[i]] = []
		cluster_dict[labels[i]].append(sort_index[i])
	# print(cluster_dict)
	key_list = []
	key_len = []
	for k in cluster_dict.keys():
		key_list.append(k)
		key_len.append(len(cluster_dict[k]))
	# print(key_list)
	# print(key_len)
	key_sort_index = np.argsort(np.array(key_len), kind="stable")
	# print(key_sort_index)
	# print(np.array(key_len)[key_sort_index])
	key_list = np.array(key_list)[key_sort_index]
	# print(key_list)
	clusters = []
	for k in key_list:
		clusters.append(cluster_dict[k])
	# print(clusters)
	cluster_len = len(clusters)
	# print(cluster_len)
	new_add = []
	j=0
	smallest_index=0
	while len(new_add) < kt:
		# new_add_ele = clusters[j][np.random.permutation(len(clusters[j]))].pop()
		# print(clusters[j])
		random.seed(100)
		random.shuffle(clusters[j])
		# print(clusters[j])
		new_add_ele = clusters[j].pop()
		# print(clusters[j])
		new_add.append(new_add_ele)
		
		if len(clusters[j])==0:
			smallest_index = j+1
			# print('sm1', smallest_index)
		if j < cluster_len-1:
			j = j+1
		else:
			j=smallest_index
		# print('j', j)
	return new_add



class CMSampling(Strategy):
	def __init__(self, train_set, test_set, handler, net, opts):
		super(CMSampling, self).__init__(train_set, test_set, handler, net, opts)
		self.total_labels = np.zeros(self.X.shape[0])
	def query(self, alpha, rd):
		"""
		alpha: a list contain weights of every domain
		rd: round
		"""
		if rd==1:
			for i in range(self.dm_num):
				self.net.logger_train.info(i)
				emb = self.get_last_embedding(self.X[self.dm==i], self.Y[self.dm==i], self.dm[self.dm==i], self.idxs_lb[self.dm==i])
				total_labels, cluster_num, dis=HAC(emb)
				self.net.logger_train.info(cluster_num)
				self.net.logger_train.info(dis)
				self.total_labels[self.dm==i]=total_labels

				# print(self.total_labels[i])
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
				probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.dm[idxs_unlabeled], self.idxs_lb[idxs_unlabeled])
				probs_sorted, idxs = probs.sort(descending=True)
				U = (probs_sorted[:, 0] - probs_sorted[:,1]).numpy() # margin
             
				kt = per_domain_num[i] # the number of new added data
				new_add = new_index(U, idxs_unlabeled, kt, self.total_labels)

				print(new_add)



			#——————————————————————————————————————————————————————————————————————————————
			new_lb_idx.append(new_add)
			print(i, per_domain_num[i], len(new_add))
			print(new_add)

		new_lb_idx = [new_lb_idx[i][j] for i in range(len(new_lb_idx)) 
		for j in range(len(new_lb_idx[i]))]

		return new_lb_idx



