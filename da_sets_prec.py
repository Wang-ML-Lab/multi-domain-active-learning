from random import sample
from re import I
from urllib.request import urlretrieve
import os
import multiprocessing
import zipfile
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import requests
import tarfile
# from unrar import rarfile


# CIFAR-10-C: wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar  
def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)
# untar("/home/dycpu1/gyh/pycharm/badge1/data/CIFAR-10-C.tar", "/home/dycpu1/gyh/pycharm/badge1/data/")
def unzip(savepath, filename):
    print(os.path.join(savepath, filename))
    zip_file = zipfile.ZipFile(os.path.join(savepath, filename))
    zip_list = zip_file.namelist() # 得到压缩包里所有文件

    for f in zip_list:
        zip_file.extract(f, savepath) # 循环解压文件到指定目录
    
    zip_file.close()
# unzip('data/digit5', 'Digit-Five.zip')

# mnist
# # train: 55000
# # test: 10000

# mnist_m
# # train: 55000
# # test: 10000
# svhn
# # train: 73257
# # test: 26032
# syn
# # train: 25000
# # test: 9000
# usps
# # train: 7438
# # test: 1860


# under "data": 1. download rarlinux-3.8.0.tar.gz 2. rar/unrar x -o- -y cross-dataset.rar ./
# rar/unrar x -o- -y office_caltech_10.rar ./
# un_rar('data/office_caltech_10.rar')
# un_rar("data/cross-dataset.rar", outdir="data/cross-dataset")
# unrar("data/cross-dataset.rar")
###

def download(url, savepath):
        """
        download file from internet
        :param url: path to download from
        :param savepath: path to save files
        :return: None
        """
        def reporthook(a, b, c):
            """
            显示下载进度
            :param a: 已经下载的数据块
            :param b: 数据块的大小
            :param c: 远程文件大小
            :return: None
            """
            print("\rdownloading: %5.1f%%" % (a * b * 100.0 / c), end="")

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        filename = os.path.basename(url)
        # 判断文件是否存在，如果不存在则下载
        if not os.path.isfile(os.path.join(savepath, filename)):
            print('Downloading data from %s' % url)
            urlretrieve(url, os.path.join(savepath, filename), reporthook=reporthook)
            print('\nDownload finished!')
        else:
            print('File already exsits!')
        # 获取文件大小
        filesize = os.path.getsize(os.path.join(savepath, filename))
        # 文件大小默认以Bytes计， 转换为Mb
        print('File size = %.2f Mb' % (filesize/1024/1024))
        if '.zip' in filename:
            unzip_path = filename[:-4]
            unzip_path = os.path.join(savepath, unzip_path)
            # print(unzip_path, os.path.join(savepath, filename))
            if not os.path.exists(unzip_path):
                # os.makedirs(unzip_path)
                '''
                基本格式：zipfile.ZipFile(filename[,mode[,compression[,allowZip64]]])
                mode：可选 r,w,a 代表不同的打开文件的方式；r 只读；w 重写；a 添加
                compression：指出这个 zipfile 用什么压缩方法，默认是 ZIP_STORED，另一种选择是 ZIP_DEFLATED；
                allowZip64：bool型变量，当设置为True时可以创建大于 2G 的 zip 文件，默认值 True；

                '''
                zip_file = zipfile.ZipFile(os.path.join(savepath, filename))
                zip_list = zip_file.namelist() # 得到压缩包里所有文件

                for f in zip_list:
                    zip_file.extract(f, savepath) # 循环解压文件到指定目录
                
                zip_file.close() # 关闭文件，必须有，释放内存
                print('\nUnzip finished!')
            else:
                print('\n Unzipped!')

def set_download(path='data', dataset="/office31", url='https://wjdcloud.blob.core.windows.net/dataset/OFFICE31.zip'):
    savepath = path+dataset
    if os.path.exists(savepath):
        print('Already download '+dataset)

    else:
        download(url, savepath)   
# set_download(path='data', dataset="/office31", url='https://wjdcloud.blob.core.windows.net/dataset/OFFICE31.zip') 
# # office31 imageCLEF digit-five/office+caltech
# set_download(path='data', dataset="/imageCLEF", url='https://wjdcloud.blob.core.windows.net/dataset/image_CLEF.zip') 

############

from sklearn.model_selection import train_test_split
def split_data(file_name):
    source_list = open(file_name).readlines()
    source_train, source_val = train_test_split(source_list, test_size=0.2)
    print(len(source_train))
    print(len(source_val))
    source_train_file_name = file_name.replace('list', 'train_list')
    source_val_file_name = file_name.replace('list', 'val_list')

    source_train_file = open(source_train_file_name, "w")
    for line in source_train:
        source_train_file.write(line)

    source_val_file = open(source_val_file_name, "w")
    for line in source_val:
        source_val_file.write(line)
# path ="data/imageCLEF/image_CLEF/image_path/"
# file_name_list = [path+item for item in ['b_list.txt', 'c_list.txt', 'i_list.txt','p_list.txt']]

# for file_name in file_name_list:
#     split_data(file_name)

# path ="data/imageCLEF/image_CLEF/image_path/"
# file_name_list = [path+item for item in ['b_list.txt', 'c_list.txt', 'i_list.txt','p_list.txt']]

# for file_name in file_name_list:
#     split_data(file_name)

def image_list(file_dir):  
    # get list about all images 
    domain_list = os.listdir(file_dir)
    class_list = os.listdir(file_dir+'/'+domain_list[0])
    class_dict ={}
    i=0
    for cls in class_list:
        class_dict[cls]=i
        i+=1
    # print(domain_list)
    # print(class_list)
    
    # L=[]
    for domain in domain_list:
        domain_list_file_name = file_dir+'/'+domain+'_list.txt'
        f=open(domain_list_file_name,'w')
        for root, dirs, files in os.walk(file_dir+'/'+domain):  
            for file in files:  
                # print(file)
                if os.path.splitext(file)[1] == '.jpg':  # 想要保存的文件格式
                    # pass
                    cls = root.split('/')[-1]
                    label=class_dict[cls]
                    target = os.path.join(root, file)+'\t'+str(label)+'\n'
                    # print(target)
                    f.write(target)
        f.close()
                    # L.append(os.path.join(root, file))  
    return 
# image_list("data/office31/OFFICE31")
# print(L[:5])
# image_list("data/office_caltech")
# file_name_list = ['data/office31/OFFICE31/amazon_list.txt', 'data/office31/OFFICE31/webcam_list.txt', 'data/office31/OFFICE31/dslr_list.txt']
# 2253
# 564
# 636
# 159
# 398
# 100

# file_name_list = ['data/office_caltech/amazon_list.txt', 'data/office_caltech/caltech_list.txt', 'data/office_caltech/dslr_list.txt', 'data/office_caltech/webcam_list.txt']
# 766
# 192
# 898
# 225
# 125
# 32
# 236
# 59
# for file_name in file_name_list:
#     split_data(file_name)


##########
def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    # path = '../' + path
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    # path = '../' + path
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def collect_data(path="/home/dycpu1/gyh/pycharm/TransCal/data", train=True, mode ='RGB'):
    if train:
        txt_name = '_train_list.txt'
    else:
        txt_name = '_val_list.txt'
    image_path = path+'/office-home/'
    domain_name =["Art", "Clipart", "Product", "Real_World"]
    image_list=[open(image_path+name_i+txt_name).readlines() for name_i in domain_name]
    dm_sample_num = [len(list_i) for list_i in image_list]
    dm = [i*np.ones(dm_sample_num[i]) for i in range(len(dm_sample_num))]
    dm_sample_num = torch.Tensor(dm_sample_num)
    dm = torch.Tensor(np.concatenate(dm, axis=0))
    X = []
    Y = []
    image_path2 = "/home/dycpu1/gyh/pycharm/TransCal"
    for i in range(len(image_list)):
        imgs_labels = make_dataset(image_list[i], labels=None)
        img_paths = [image_path2+i[0] for i in imgs_labels]
        labels = [i[1] for i in imgs_labels]
        X +=img_paths
        Y +=labels
    Y = torch.from_numpy(np.array(Y)).long()
    X = np.array(X)      
    return X, Y, dm, dm_sample_num # X, Y: numpy, dm, dm_sample_num:torch.Tensor

def collect_office31(path="data", train=True, mode ='RGB'):
    if train:
        txt_name = '_train_list.txt'
    else:
        txt_name = '_val_list.txt'
    image_path = path+'/office31/OFFICE31/'
    domain_name =["amazon", "dslr", "webcam"]
    image_list=[open(image_path+name_i+txt_name).readlines() for name_i in domain_name]
    dm_sample_num = [len(list_i) for list_i in image_list]
    dm = [i*np.ones(dm_sample_num[i]) for i in range(len(dm_sample_num))]
    dm_sample_num = torch.Tensor(dm_sample_num)
    dm = torch.Tensor(np.concatenate(dm, axis=0))
    X = []
    Y = []

    for i in range(len(image_list)):
        imgs_labels = make_dataset(image_list[i], labels=None)
        img_paths = [i[0] for i in imgs_labels]
        labels = [i[1] for i in imgs_labels]
        X +=img_paths
        Y +=labels
    Y = torch.from_numpy(np.array(Y)).long()
    X = np.array(X)      
    return X, Y, dm, dm_sample_num # X, Y: numpy, dm, dm_sample_num:torch.Tensor
# X, Y, dm, dm_sample_num = collect_office31()
# print(X[:5])

def collect_ofct(path="data", train=True, mode ='RGB'):
    if train:
        txt_name = '_train_list.txt'
    else:
        txt_name = '_val_list.txt'
    image_path = path+'/office_caltech/'
    domain_name =["amazon", 'caltech', "dslr", "webcam"]
    image_list=[open(image_path+name_i+txt_name).readlines() for name_i in domain_name]
    dm_sample_num = [len(list_i) for list_i in image_list]
    dm = [i*np.ones(dm_sample_num[i]) for i in range(len(dm_sample_num))]
    dm_sample_num = torch.Tensor(dm_sample_num)
    dm = torch.Tensor(np.concatenate(dm, axis=0))
    X = []
    Y = []

    for i in range(len(image_list)):
        imgs_labels = make_dataset(image_list[i], labels=None)
        img_paths = [i[0] for i in imgs_labels]
        labels = [i[1] for i in imgs_labels]
        X +=img_paths
        Y +=labels
    Y = torch.from_numpy(np.array(Y)).long()
    X = np.array(X)      
    return X, Y, dm, dm_sample_num # X, Y: numpy, dm, dm_sample_num:torch.Tensor
# X, Y, dm, dm_sample_num = collect_ofct()
# print(X.shape)

def collect_clef(path="data", train=True, mode ='RGB'):
    if train:
        txt_name = '_train_list.txt'
    else:
        txt_name = '_val_list.txt'
    image_path = path+'/imageCLEF/image_CLEF/image_path/'
    domain_name =["b", "c", "i", "p"]
    image_list=[open(image_path+name_i+txt_name).readlines() for name_i in domain_name]
    dm_sample_num = [len(list_i) for list_i in image_list]
    dm = [i*np.ones(dm_sample_num[i]) for i in range(len(dm_sample_num))]
    dm_sample_num = torch.Tensor(dm_sample_num)
    dm = torch.Tensor(np.concatenate(dm, axis=0))
    X = []
    Y = []
    image_path2 = path+'/imageCLEF/image_CLEF/'
    for i in range(len(image_list)):
        imgs_labels = make_dataset(image_list[i], labels=None)
        labels = [i[1] for i in imgs_labels]
        if i<2:
            img_paths = [image_path2+i[0] for i in imgs_labels]
            img_paths = [img_paths[i].replace(img_paths[i].split('/')[-2], str(labels[i])) for i in range(len(img_paths))]
        else:
            img_paths = [image_path2+imgs_labels[i][0][:2] +str(labels[i])+'/'+imgs_labels[i][0][2:] for i in range(len(imgs_labels))]

        X +=img_paths
        Y +=labels
    Y = torch.from_numpy(np.array(Y)).long()
    X = np.array(X)      
    return X, Y, dm, dm_sample_num # X, Y: numpy, dm, dm_sample_num:torch.Tensor
# X, Y, dm, dm_sample_num = collect_clef()
# print(X[:5])
# for i in X:
#     print(i)
# print(X.shape, Y.shape, dm.shape, dm_sample_num) (1920,) torch.Size([1920]) torch.Size([1920]) tensor([480., 480., 480., 480.])
def split_val_test(te):
    X, Y, dm, dm_sample_num = te
    index = np.array([i for i in range(dm.size(0))])
    sample_num=1725
    dm_idx = np.array([int(dm[i].numpy()) for i in range(dm.size(0))],dtype=int)
    index_val = []
    index_test = []
    dm_sample_num_val = []
    dm_sample_num_test = []
    np.random.seed(1234)
    for i in range(6):
        index_i = index[dm_idx==i]
        index_i = index_i[np.random.permutation(len(index_i))]
        index_i_val, index_i_test = index_i[:sample_num], index_i[sample_num:]
        index_val.append(index_i_val)
        index_test.append(index_i_test)

        dm_sample_num_val.append(len(index_i_val))
        dm_sample_num_test.append(len(index_i_test))
    index_val = np.concatenate(index_val, axis=0)
    index_test = np.concatenate(index_test, axis=0)
    
    val = X[index_val], Y[index_val], dm[index_val], torch.Tensor(dm_sample_num_val)
    test = X[index_test], Y[index_test], dm[index_test], torch.Tensor(dm_sample_num_test)


    return val, test

class DataHandler_da(Dataset):
    def __init__(self, X, Y, dm, lb, transform=None, mode = 'RGB'):
        self.X = X
        self.Y = Y
        self.dm = dm
        self.lb = lb
        self.transform = transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        x, y, dm, lb = self.X[index], self.Y[index], self.dm[index], self.lb[index]
        x = self.loader(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y, dm, lb, index

    def __len__(self):
        return len(self.X)


class DataHandler_da_ft(Dataset):
    def __init__(self, X, Y, dm, lb, transform=None, mode = 'RGB'):
        self.X = X
        self.Y = Y
        self.dm = dm
        self.lb = lb
        # self.transform = transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        x, y, dm, lb = self.X[index], self.Y[index], self.dm[index], self.lb[index]
        return x, y, dm, lb, index

    def __len__(self):
        return len(self.X)
