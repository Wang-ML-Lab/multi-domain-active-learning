import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from copy import deepcopy
import pdb
import resnet
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
class Strategy:
    def __init__(self, train_set, test_set, handler, net, opts):
        self.dm_num = opts.domain_num
        self.X, self.Y, self.dm, self.idxs_lb = train_set
        self.n_init =opts.nStart
        self.n_query = opts.nQuery
        self.net = net
        self.handler = handler
        self.args = opts.args
        self.n_pool = len(self.Y)
        self.opt = opts
        
        # print(type(self.X), type(self.Y), type(self.dm), type(self.idxs_lb))
        if opts.data=='MNIST' or opts.data=='office-home' or opts.data=='cct20':
            X_te, Y_te, dm_te, idxs_lb_te = test_set
            self.test_set = DataLoader(self.handler(X_te, Y_te, dm_te, torch.Tensor(idxs_lb_te).long(), transform=self.args['transformTest']),
                                shuffle=False, **self.args['loader_te_args'])
        elif opts.data=='domainnet':
            X_te, Y_te, dm_te, idxs_lb_te, X_val, Y_val, dm_val, idxs_lb_val = test_set
            self.test_set = DataLoader(self.handler(X_val, Y_val, dm_val, torch.Tensor(idxs_lb_val).long(), transform=self.args['transformTest']),
                                shuffle=False, **self.args['loader_te_args']), \
                                    DataLoader(self.handler(X_te, Y_te, dm_te, torch.Tensor(idxs_lb_te).long(), transform=self.args['transformTest']),
                                shuffle=False, **self.args['loader_te_args'])
        else:
            X_te, Y_te, dm_te, idxs_lb_te = test_set
            self.test_set = DataLoader(self.handler(X_te, Y_te, dm_te, torch.Tensor(idxs_lb_te).long(), transform=self.args['transformTest']),
                                shuffle=False, **self.args['loader_te_args'])
        
    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
    
    def train(self, rd):
        # print('T',self.args['transform'])
        n_epoch = self.args['n_epoch']
        print("number of active labels",sum(self.idxs_lb))
### prepare datasets-----------------------------
        train_set = DataLoader(
        self.handler(self.X, self.Y, self.dm, torch.Tensor(self.idxs_lb).long(), transform=self.args['transform']), 
        shuffle=True, **self.args['loader_tr_args'])
        
        self.idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        train_label_set = DataLoader(
            self.handler(
        self.X[self.idxs_train], 
        self.Y[self.idxs_train], 
        self.dm[self.idxs_train],
        torch.Tensor(self.idxs_lb[self.idxs_train]).long(), 
        transform=self.args['transform']
        ), shuffle=True, **self.args['loader_label_args'])
        

        # if rd==0:
        if True:
            self.loader_ref = DataLoader(self.handler(
                self.X[self.idxs_train], 
                self.Y[self.idxs_train], 
                self.dm[self.idxs_train],
                torch.Tensor(self.idxs_lb[self.idxs_train]).long(), 
                transform=self.args['transformTest']),
                            shuffle=False, **{'batch_size': 1, 'num_workers': 1}) # reference
### prepare datasets-----------------------------
        
        ### train and test model
        self.net.train_process(n_epoch, rd, train_set, train_label_set, self.test_set)
        
        
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        for batch_idx, (x, y, dm, idxs) in enumerate(loader_tr):
            x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
            optimizer.zero_grad()
            out, e1 = self.clf(x, dm)
            loss = F.cross_entropy(out, y)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.get_parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset.X)

    def predict(self, X, Y, dm, rd):
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, torch.Tensor(dm), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, torch.Tensor(dm), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        
        self.clf.load(rd)
        self.clf.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, dm, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                out, e1 = self.clf(x, dm)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X, Y, dm, idxs_lb):
        loader_te = DataLoader(self.handler(X, Y, dm, torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.net.load()
        clf = self.net.netE
        clf.eval()
        # probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        probs = []
        # print(probs.size())
        with torch.no_grad():
            for x, y, dm, _, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                out, e1 = clf(x, dm)
                if len(out.shape)==1:
                    out = out.view(1, -1)
                out =out[:, :self.opt.class_num]

                prob = F.softmax(out, dim=1)
                # print(prob.shape)
                probs.append(prob.cpu().data)
        
        return torch.cat(probs, dim=0)

    def predict_emb(self, X, Y, dm, idxs_lb):
        loader_te = DataLoader(self.handler(X, Y, dm, torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.net.load()
        encoder = self.net.netE
        encoder.eval()
        
        # probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        out_list = []
        # print(probs.size())
        with torch.no_grad():
            for x, y, dm, _, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                _, e = encoder(x, dm)
                out_list.append(e.cpu().data)
        return torch.cat(out_list, dim=0)

    def predict_gan_prob(self, X, Y, dm, idxs_lb):
        loader_te = DataLoader(self.handler(X, Y, dm, torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.net.load()
        dis = self.net.netD
        encoder = self.net.netE
        dis.eval()
        encoder.eval()
        
        # probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        out_list = []
        # print(probs.size())
        with torch.no_grad():
            for x, y, dm, _, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                _, e = encoder(x, dm)
                out = dis(e, dm)
        #         if len(out.shape)==1:
        #             out = out.view(1, -1)
        #         out =out[:, :self.opt.class_num]

        #         prob = F.softmax(out, dim=1)
        #         # print(prob.shape)
        #         probs.append(prob.cpu().data)
        
        # return torch.cat(probs, dim=0)
                out_list.append(out.view(-1).cpu().data)
        return torch.cat(out_list, dim=0)

    def predict_free_energy(self, X, Y, dm, idxs_lb):
        loader_te = DataLoader(self.handler(X, Y, dm, torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.net.load()
        clf = self.net.netE
        clf.eval()
        # probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        engs = []
        # print(probs.size())
        with torch.no_grad():
            for x, y, dm, _, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                out, e1 = clf(x, dm)
                if len(out.shape)==1:
                    out = out.view(1, -1)
                out =out[:, :self.opt.class_num]

                eng = -torch.log(torch.sum(torch.exp(out), dim=1)) # free energy
                # print(prob.shape)
                engs.append(eng.cpu().data)
        
        return torch.cat(engs, dim=0)
    
    def predict_joint_energy(self, X, Y, dm, idxs_lb):
        loader_te = DataLoader(self.handler(X, Y, dm, torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.net.load()
        clf = self.net.netE
        clf.eval()
        # probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        engs = []
        # print(probs.size())
        with torch.no_grad():
            for x, y, dm, _, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                out, e1 = clf(x, dm)
                if len(out.shape)==1:
                    out = out.view(1, -1)
                out =out[:, :self.opt.class_num]

                # print(prob.shape)
                engs.append(out.cpu().data)
        
        return torch.cat(engs, dim=0)

    def predict_influ(self, X, Y, dm, idxs_lb):
        
        self.ref_grad()
        self.ref_s_test()

        loader_te = DataLoader(self.handler(X, Y, dm, torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **{'batch_size': 1, 'num_workers': 1})
        self.net.load()
        clf = self.net.netE.cpu()
        clf.eval()
        params = [p for p in clf.get_parameters() if p.requires_grad and len(p.size()) != 1]
        
        influence_list = self.calc_influence_funtion(clf, loader_te, params, self.s_test_list)
        # print("len(influence_list)", len(influence_list))
        return torch.Tensor(influence_list)
    
    def calc_influence_funtion(self, model, test_loader, params, s_test):
        """ non-distributed version of calculation of influence value.

        Args:
            model (nn.Module): Model to calculate the loss.
            test_loader (nn.Dataloader): Pytorch data loader with the unlabeled data.
            params: the parameters of model to calculate the gradients.
            s_test: the s_test value.
        Returns:
            list [tensor]: The s_test results. The list contain the gradient calculated for per parameter.
        """

        loss_fun = SoftCrossEntropyLoss() #.cuda()
        influence_list = []
        # idxes_list = []
        # s_test = torch.cat([ele.view(-1).cuda() for ele in s_test])
        s_test = torch.cat([ele.view(-1) for ele in s_test])
        # print('s_test.shape', s_test.shape)
        for x, y, dm, _, idxs in test_loader:
            # x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
            preds, e1 = model(x, dm)
            preds = preds[:self.opt.class_num]
            pseudo_label = torch.nn.functional.softmax(preds, dim=0).argmax().unsqueeze(dim=0)
            # print(pseudo_label)
            labels_one_hot = torch.nn.functional.one_hot(pseudo_label, self.opt.class_num).float() #.cuda()
            loss = loss_fun(preds, labels_one_hot)
            grad_result = grad(loss, params)

            grad_result = [ele.detach().view(-1) for ele in grad_result]
            grad_concat = torch.cat(grad_result)
            # print('grad_concat.shape', grad_concat.shape)
            # result = - torch.sum(grad_concat * s_test).cpu().numpy() / len(test_loader.dataset)
            result = - torch.sum(grad_concat * s_test).numpy() / len(test_loader.dataset)
            influence_list.append(result)
            # idxes_list.append(idxs)

        return influence_list

    def ref_grad(self):
        self.net.load()
        clf = self.net.netE.cpu()
        clf.eval()
        params = [p for p in clf.get_parameters() if p.requires_grad and len(p.size()) != 1]
        loss_fun = SoftCrossEntropyLoss() #.cuda()
        grad_list = []
        
        for x, y, dm, _, idxs in self.loader_ref:
            # x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
            labels_one_hot = torch.nn.functional.one_hot(y, self.opt.class_num).float() #.cuda()
            # print("y.shape, labels_one_hot.shape", y.shape, labels_one_hot.shape)
            preds, e1 = clf(x, dm)
            preds = preds[:self.opt.class_num]
            loss = loss_fun(preds, labels_one_hot)
            grad_result = grad(loss, params)
            for ele in grad_result:
                ele.detach()
            grad_list.append(grad_result)

        self.val_sum_grad = []
        for i in range(len(grad_list[0])):
            per_param_grad = [torch.unsqueeze(per_img_grad[i], dim=0) for per_img_grad in grad_list]
            per_param_grad = torch.cat(per_param_grad, dim=0)
            per_param_grad = torch.sum(per_param_grad, dim=0)
            self.val_sum_grad.append(per_param_grad)
        return self.val_sum_grad 

    def ref_s_test(self):
        
        self.net.load()
        clf = self.net.netE.cpu()
        clf.eval()
        params = [p for p in clf.get_parameters() if p.requires_grad and len(p.size()) != 1]

        loss_val_mul_hessian_reverse = self.s_test(self.val_sum_grad, clf, params, self.loader_ref)
        self.s_test_list=[]
        for i in range(len(loss_val_mul_hessian_reverse[0])):
            per_param_grad = [torch.unsqueeze(per_node_grad[i], dim=0) for per_node_grad in loss_val_mul_hessian_reverse]
            num_of_node = len(per_param_grad)
            per_param_grad = torch.cat(per_param_grad, dim=0)
            per_param_grad = torch.sum(per_param_grad, dim=0)
            per_param_grad /= num_of_node
            self.s_test_list.append(per_param_grad)
        
    
    def s_test(self, val_grad, model, params, train_data_loader, damp=0.01, scale=10000.0, img_num=25):
        """ non-distributed version of calculation of s_test.
        s_test is the Inverse Hessian Vector Product.

        Args:
            val_grad: the sum of the gradient of whole validation set.
            model (nn.Module): Model to calculate the loss.
            params: the parameters of model to calculate the gradients.
            data_loader (nn.Dataloader): Pytorch data loader with the train data.
            damp: float, dampening factor.
            scale: float, scaling factor, a list, you can set up a specific scale for each parameter.
            img_num: use how many img to calculate s_test per node.

        Returns:
            list [tensor]: The s_test results. The list contain the gradient calculated for per parameter.
        """
        loss_fun = SoftCrossEntropyLoss() #.cuda()
        h_estimate = val_grad.copy()
        h_estimate_list = []
        cur_iter = 0
        for x, y, dm, _, idxs in train_data_loader:
            # x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
            labels_one_hot = torch.nn.functional.one_hot(y, self.opt.class_num).float() #.cuda()
            preds, e1 = model(x, dm)
            preds = preds[:self.opt.class_num]
            loss = loss_fun(preds, labels_one_hot)

            hv = self.hessian_vec_prod(loss, params, h_estimate)

            h_estimate_updated = []
            for _v, _h_e, _hv in zip(val_grad, h_estimate, hv):
                if True in torch.isnan(_hv):
                    h_estimate_updated.append(_v)
                else:
                    h_estimate_updated.append(_v + (1 - damp) * _h_e - _hv / scale)
            h_estimate = h_estimate_updated

            if cur_iter % 100 == 0:
                print(cur_iter)
            # if cur_iter >= img_num*4:
            #     break
            if cur_iter % img_num == img_num - 1:
                h_estimate_list.append(h_estimate)
                h_estimate = val_grad.copy()
            cur_iter +=1
        return h_estimate_list
    
    def hessian_vec_prod(self, loss, params, h_estimate):
        """Multiply the Hessians of y and w by v.
        Uses a backprop-like approach to compute the product between the Hessian
        and another vector efficiently, which even works for large Hessians.
        Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
        which evaluates to the same values as (A + A.t) v.

        Arguments:
            loss: The loss from each training data
            params: The parameters of model which need to calculate the gradients
            h_estimate: the hessian_vec_prod in the last iteration

        Returns:
            return_grads: list of torch tensors, contains product of Hessian and v.

        Raises:
            ValueError: `y` and `w` have a different length."""
        if len(params) != len(h_estimate):
            raise(ValueError("w and v must have the same length."))

        # First backprop
        first_grads = grad(loss, params, retain_graph=True, create_graph=True, allow_unused=True)
        
        # elemwise_products = [torch.unsqueeze(torch.sum(grad_elem.cuda() * v_elem.cuda()), dim=0) for grad_elem, v_elem in zip(first_grads, h_estimate)]
        elemwise_products = [torch.unsqueeze(torch.sum(grad_elem * v_elem), dim=0) for grad_elem, v_elem in zip(first_grads, h_estimate)]
        
        elemwise_products = torch.cat(elemwise_products, dim=0)
        per_param_grad = torch.sum(elemwise_products, dim=0)

        # Second backprop
        return_grads = grad(per_param_grad, params, create_graph=True, allow_unused=True)
        return_grads = [ele.detach() for ele in return_grads]

        return return_grads

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding

    # discriminator's gradient embedding (assumes cross-entropy loss)
    def get_dis_grad_embedding(self, X, Y, dm, idxs_lb):

        self.net.load()
        dis = self.net.netD
        encoder = self.net.netE
        dis.eval()
        encoder.eval()
        embDim = 512 # emb size of last layer of discriminator
        
        nLab = 2
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, torch.Tensor(dm), torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, dm, _, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                _, e = encoder(x, dm)
                cout, out = dis.get_emb(e, dm)
                if len(out.shape)==1:
                    cout = cout.view(1, -1)
                    out = out.view(1, -1)
                out = out.data.cpu().numpy()
                cout = cout.data.cpu().numpy()
                # print("cout.shape",cout.shape)
                batchProbs = np.concatenate((1.0-cout, cout), axis=1)
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, dm, idxs_lb):
        self.net.load()
        model = self.net.netE
        embDim = self.opt.nh # emb size of last layer of classifier
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, torch.Tensor(dm), torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, dm, _, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                cout, out = model.get_emb(x, dm)
                if len(out.shape)==1:
                    cout = cout.view(1, -1)
                    out = out.view(1, -1)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout[:, :self.opt.class_num], dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)

    def get_last_embedding(self, X, Y, dm, idxs_lb):
        self.net.load()
        model = self.net.netE
        embDim = 256 # emb size of last layer of classifier
        model.eval()
        embedding = []
        loader_te = DataLoader(self.handler(X, Y, torch.Tensor(dm), torch.Tensor(idxs_lb).long(), transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, dm, _, idxs in loader_te:
                x, y, dm = Variable(x.cuda()), Variable(y.cuda()), Variable(dm.cuda())
                cout, out = model.get_emb(x, dm)
                out = out.data.cpu().numpy()
                embedding.append(out)
        embeding = np.concatenate(embedding, axis=0)
        return embeding

class SoftCrossEntropyLoss(torch.nn.Module):
    """SoftCrossEntropyLoss (useful for label smoothing and mixup).
    Identical to torch.nn.CrossEntropyLoss if used with one-hot labels."""

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        # print('y.shape, x.shape, torch.nn.functional.log_softmax(x, -1).shape', y.shape, x.shape, torch.nn.functional.log_softmax(x, -1).shape)
        loss = -y * torch.nn.functional.log_softmax(x, -1)

        return torch.sum(loss) / x.shape[0]