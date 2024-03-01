import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from da_sets_prec import *
# from domainnet_prec import *
# from cct_prec import *
def get_dataset(name, path, rotate_angle=None, opts=None):
    if name == 'MNIST':
        return get_MNIST(path, rotate_angle)
    elif name == 'office-home':
        return get_of_ft(opts)
        # return get_of()
    elif name == 'office31':
        return get_of_ft(opts)
        # return get_of31()
    elif name == 'office_caltech':
        return get_of_ft(opts)
        # return get_ofct()
    elif name == 'imageCLEF':
        if opts.imageCLEF_single==True:
            return get_of_ft_single(opts)
        return get_of_ft(opts)
    
        
        # return get_clef()
        
            
    elif name == 'cct20':
        return get_cct20_ft(opts)
        # return get_cct20(path)
    elif name == 'domainnet':
        return get_dn_ft(opts)
        # return get_domainnet(path)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)

def rotate_func(X, Y, rotate_angle):
    from torchvision.transforms.functional import rotate
    n_pool =Y.size(0)
    domain_num = int(len(rotate_angle)/2)
    dm_idx_all = np.zeros(n_pool) # domain index: 0 to domain_num-1
    sample_idxs = np.arange(n_pool) # image index: n is the number of images
    np.random.shuffle(sample_idxs)
    per_domain_num = int(n_pool/domain_num) # 向下取整
    for i in range(domain_num):
        if i==domain_num-1:
            dm_idx_all[sample_idxs[i*per_domain_num:]] = i
        else:
            # print(i)
            # print(dm_idx_all[sample_idxs[i*per_domain_num:(i+1)*per_domain_num]])
            dm_idx_all[sample_idxs[i*per_domain_num:(i+1)*per_domain_num]] = i # 取前init个来标注
            

    img_all =[]
    target_all = []
    angle_all = []
    index_all = []
    # dm_idx_all = []
    for index in range(len(X)):
        d_idx = dm_idx_all[index]
        rot_min, rot_max = rotate_angle[int(d_idx*2)],rotate_angle[int(d_idx*2+1)]
        img, target = X[index], int(Y[index])
        # img = Image.fromarray(img.numpy(), mode='L')
        angle = np.random.rand() * (rot_max - rot_min) + rot_min
        img1 = img.view(-1, 28, 28)
        # print(type(img), img.shape)
        img = rotate(img1, angle)
        # print(img1==img)
        # img = transforms.ToTensor()(img)
        # print(type(img))
        img_all.append(img)
        target_all.append(target)
        angle_all.append(angle)

    X = torch.from_numpy(np.concatenate(img_all, axis=0))
    Y = torch.from_numpy(np.array(target_all)).long()
    angle_all = np.array(angle_all)
    index_all = np.arange(n_pool)
    # active_all = np.zeros_like(index_all)
    dm_sample_num = []
    dm_sample_index = []
    for i in range(domain_num):
        dm_sample_num.append(len(index_all[dm_idx_all==i])) # number of every domain
        dm_sample_index.append(index_all[dm_idx_all==i]) # index of every domain
    dm_sample_num = torch.Tensor(dm_sample_num)
    Y = torch.Tensor(Y.numpy()).long()
    dm_idx_all = torch.Tensor(dm_idx_all)
    return X, Y, dm_idx_all, dm_sample_num, dm_sample_index, angle_all

def get_of():
    tr = collect_data(train=True)
    te = collect_data(train=False)
    return tr, te
def get_of31():
    tr = collect_office31(train=True)
    te = collect_office31(train=False)
    return tr, te

def get_ofct():
    tr = collect_ofct(train=True)
    te = collect_ofct(train=False)
    return tr, te

def get_clef():
    #imageclef
    tr = collect_clef(train=True)
    te = collect_clef(train=False)
    return tr, te

def get_cct20(path):
    return collect_cct20(path)

def get_cct20_ft(opts):
    """feature version"""
    save_resnet_path = opts.path+'/'+'cct/20_'+opts.resnet_name+'/'
    X_tr = np.load(os.path.join(save_resnet_path, 'X_tr.npy'))
    Y_tr = np.load(os.path.join(save_resnet_path, 'Y_tr.npy'))
    dm_tr = np.load(os.path.join(save_resnet_path, 'dm_tr.npy'))
    dm_sample_num_tr = np.load(os.path.join(save_resnet_path, 'dm_sample_num_tr.npy'))
    tr = torch.Tensor(X_tr), torch.Tensor(Y_tr).long(), torch.Tensor(dm_tr), torch.Tensor(dm_sample_num_tr)
    
    X_te = np.load(os.path.join(save_resnet_path, 'X_te.npy'))
    Y_te = np.load(os.path.join(save_resnet_path, 'Y_te.npy'))
    dm_te = np.load(os.path.join(save_resnet_path, 'dm_te.npy'))
    dm_sample_num_te = np.load(os.path.join(save_resnet_path, 'dm_sample_num_te.npy'))
    te = torch.Tensor(X_te), torch.Tensor(Y_te).long(), torch.Tensor(dm_te), torch.Tensor(dm_sample_num_te)
    return tr, te

def get_of_ft(opts):
    """feature version"""
    save_resnet_path = opts.path+'/'+opts.data+'/'+opts.resnet_name+'/'
    X_tr = np.load(os.path.join(save_resnet_path, 'X_tr.npy'))
    Y_tr = np.load(os.path.join(save_resnet_path, 'Y_tr.npy'))
    dm_tr = np.load(os.path.join(save_resnet_path, 'dm_tr.npy'))
    dm_sample_num_tr = np.load(os.path.join(save_resnet_path, 'dm_sample_num_tr.npy'))
    tr = torch.Tensor(X_tr), torch.Tensor(Y_tr).long(), torch.Tensor(dm_tr), torch.Tensor(dm_sample_num_tr)
    
    X_te = np.load(os.path.join(save_resnet_path, 'X_te.npy'))
    Y_te = np.load(os.path.join(save_resnet_path, 'Y_te.npy'))
    dm_te = np.load(os.path.join(save_resnet_path, 'dm_te.npy'))
    dm_sample_num_te = np.load(os.path.join(save_resnet_path, 'dm_sample_num_te.npy'))
    te = torch.Tensor(X_te), torch.Tensor(Y_te).long(), torch.Tensor(dm_te), torch.Tensor(dm_sample_num_te)
    return tr, te

def get_of_ft_single(opts):
    """feature version"""
    save_resnet_path = opts.path+'/'+opts.data+'/'+opts.resnet_name+'/'
    X_tr = np.load(os.path.join(save_resnet_path, 'X_tr.npy'))
    Y_tr = np.load(os.path.join(save_resnet_path, 'Y_tr.npy'))
    dm_tr = np.load(os.path.join(save_resnet_path, 'dm_tr.npy'))
    dm_sample_num_tr = np.load(os.path.join(save_resnet_path, 'dm_sample_num_tr.npy'))
    dm_tr = torch.Tensor(dm_tr)
    tr = torch.Tensor(X_tr)[dm_tr.long()==0], torch.Tensor(Y_tr).long()[dm_tr.long()==0], dm_tr[dm_tr.long()==0], torch.Tensor([dm_sample_num_tr[0]])
    
    X_te = np.load(os.path.join(save_resnet_path, 'X_te.npy'))
    Y_te = np.load(os.path.join(save_resnet_path, 'Y_te.npy'))
    dm_te = np.load(os.path.join(save_resnet_path, 'dm_te.npy'))
    dm_sample_num_te = np.load(os.path.join(save_resnet_path, 'dm_sample_num_te.npy'))
    dm_te = torch.Tensor(dm_te)
    te = torch.Tensor(X_te)[dm_te.long()==0], torch.Tensor(Y_te).long()[dm_te.long()==0], dm_te[dm_te.long()==0], torch.Tensor([dm_sample_num_te[0]])
    return tr, te

def get_dn_ft(opts):
    """feature version"""
    save_resnet_path = opts.path+'/'+opts.data+'/'+opts.resnet_name+'/'
    X_tr = np.load(os.path.join(save_resnet_path, 'X_tr.npy'))
    Y_tr = np.load(os.path.join(save_resnet_path, 'Y_tr.npy'))
    dm_tr = np.load(os.path.join(save_resnet_path, 'dm_tr.npy'))
    dm_sample_num_tr = np.load(os.path.join(save_resnet_path, 'dm_sample_num_tr.npy'))
    tr = torch.Tensor(X_tr), torch.Tensor(Y_tr).long(), torch.Tensor(dm_tr), torch.Tensor(dm_sample_num_tr)
    
    X_val = np.load(os.path.join(save_resnet_path, 'X_val.npy'))
    Y_val = np.load(os.path.join(save_resnet_path, 'Y_val.npy'))
    dm_val = np.load(os.path.join(save_resnet_path, 'dm_val.npy'))
    dm_sample_num_val = np.load(os.path.join(save_resnet_path, 'dm_sample_num_val.npy'))
    val = torch.Tensor(X_val), torch.Tensor(Y_val).long(), torch.Tensor(dm_val), torch.Tensor(dm_sample_num_val)

    X_te = np.load(os.path.join(save_resnet_path, 'X_te.npy'))
    Y_te = np.load(os.path.join(save_resnet_path, 'Y_te.npy'))
    dm_te = np.load(os.path.join(save_resnet_path, 'dm_te.npy'))
    dm_sample_num_te = np.load(os.path.join(save_resnet_path, 'dm_sample_num_te.npy'))
    te = torch.Tensor(X_te), torch.Tensor(Y_te).long(), torch.Tensor(dm_te), torch.Tensor(dm_sample_num_te)
    return tr, val, te 


def get_MNIST(path, rotate_angle):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    # print(X_tr.shape, Y_tr.shape)
    tr = rotate_func(X_tr, Y_tr, rotate_angle)
    X_te = raw_te.data
    Y_te = raw_te.targets
    te = rotate_func(X_te, Y_te, rotate_angle)
    return tr, te

def get_d5(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    # print(X_tr.shape, Y_tr.shape)
    tr = rotate_func(X_tr, Y_tr, rotate_angle)
    X_te = raw_te.data
    Y_te = raw_te.targets
    te = rotate_func(X_te, Y_te, rotate_angle)
    return tr, te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te
    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name == 'MNIST':
        # print('3')
        return DataHandler3
    elif name == 'office-home':
        # print('3')
        return DataHandler_da_ft
    elif name == 'imageCLEF':
        # print('3')
        return DataHandler_da_ft
        # return DataHandler_da
    elif name == 'domainnet':
        # print('3')
        return DataHandler_da_ft
    elif name in ['office31', 'office_caltech']:
        return DataHandler_da_ft
    elif name == 'cct20':
        # print('3')
        return DataHandler_da_ft
        # return DataHandler_da
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    else:
        # print('4')
        return DataHandler4

# class DataHandler5(Dataset):
#     def __init__(self, X_Y, dm, lb, transform=None):
#         self.X = X
#         self.Y = Y
#         self.dm = dm
#         self.lb = lb
#         self.transform = transform

#     def __getitem__(self, index):
#         x, y, dm, lb = self.X[index].numpy(), self.Y[index], self.dm[index], self.lb[index]
#         if self.transform is not None:
#             x = Image.fromarray(x)
#             x = self.transform(x)
#         return x, y, dm, lb, index

#     def __len__(self):
#         return len(self.X)

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, dm, lb, transform=None):
        self.X = X
        self.Y = Y
        self.dm = dm
        self.lb = lb
        self.transform = transform

    def __getitem__(self, index):
        x, y, dm, lb = self.X[index].numpy(), self.Y[index], self.dm[index], self.lb[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, dm, lb, index

    def __len__(self):
        return len(self.X)


class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)
