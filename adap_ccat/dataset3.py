from glob import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
from PIL import Image
import cv2
import os
import numpy as np
from os.path import join as osj
from skimage.restoration import denoise_wavelet
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.utils import save_image
import random
import natsort

def sorting_truncated(fn):
    len1=len(fn)
       
    fns = []
    basename = '/'.join(fn[0].split('/')[:-1])

    for i in range(len(fn)):
        path=osj(basename, str(i)+'.jpg')
        if os.path.isfile(path):
            fns.append(path)
#         else:
#             print('Something wrong!!', path)
    return fns
def check_fn(filename):
    assert os.path.isfile(filename)


# def get_train_augmentation(image_size, crop_size):
#     # TODO order of resize and randcrop?
#     return V.Compose([
#         V.Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
#         V.Resize(image_size, interpolation=1, always_apply=True, p=1.0),
#         V.RandomCrop(crop_size, p=1.0),
#         # V.GaussianNoise(var_limit=(0, 5), p=0.5),
#         # V.RandomBrightnessContrast(p=0.5)
#         # V.RandomGamma(gamma_limit=(80, 120), p=0.5),
#     ], p=1.0)

class dataset_DFD(torch.utils.data.Dataset):
    
    def __init__(self, local_rank, args, mode='train', file_list=None, root=None):
        super(dataset_DFD, self).__init__()

        self.mode        = mode
        self.root        = args.dataset_dir if mode!='test' and mode!='finetuning' else args.dataset_dir_test
        self.FRR         = args.FRR
        self.FREQ        = args.FREQ
        self.seq_len     = []
        self.fns         = []
        self.image_size  = args.image_size
        self.crop_size   = args.crop_size
        self.offsets     = (args.image_size-args.crop_size)//2
        self.random_crop = True
        self.marginal    = args.marginal if mode!='test' else 0
        self.label       = []
        self.local_rank  = local_rank
        self.test_aug    = args.test_aug
        self.max_det     = args.max_det
        self.FSet        = [int(v) for v in args.MultiFREQ.split(',')] if args.MultiFREQ is not None else None
        filename         = args.train_file if mode!='test' else args.test_file
        self.maxFREQ     = args.FRR if (self.FSet is None or mode=='test') else max(self.FSet)
        self.centerCrop  = args.centerCrop/100.0
        self.twoStream   = args.twoStream
        self.allRandRoate= args.allRandRoate
        self.singleAug   = args.singleAug
        self.evalPerformance= args.evalPerformance

        self.image_list = []
        if filename == 0:

            self.image_list = []
            for lab, folder1 in enumerate(['neg', 'pos']): # neg: 0, pos: 1
                folders2 = glob(osj(self.root, folder1, '*'))
                for folder2 in folders2:
                    fn_list = glob(osj(self.root, folder1, folder2, '*.jpg'))
                    fn_list = natsort.natsorted(fn_list)
                    #fn_list = sorting_truncated(fn_list)
                    if len(fn_list) < self.FRR:
                        print('Pass the filename', osj(folder1, folder2),
                              'due to inefficient number of training samples: target:', self.FRR,
                              'source:', len(fn_list))

                    self.label.append(lab)
                    self.image_list.append(fn_list)
                    self.seq_len.append(len(fn_list))
                    self.fns.append(osj(self.root, folder1, folder2))
        #self.label = [0, 1, 0 , 1, 1, 1,]
        # self.image_list = [[~~~/0.jpg,~~~/1.jpg , , ,~~~/51.jpg], [] , [] ]

        else:
            print('Preparing the file from', filename)
            with open(filename, 'r') as fp:
                data = fp.readlines()
                for line in data:
                    fn, lab = line.split(' ')
                    fn_list = glob(osj(self.root, fn, '*.jpg'))
                    fn_list = natsort.natsorted(fn_list)
                    #fn_list = sorting_truncated(fn_list)

                    if len(fn_list) < self.FRR:
                        print('Pass the filename', fn, 'due to inefficient number of training samples: target:',
                              self.FRR, 'source:', len(fn_list))
                    #

                    self.label.append(int(lab.strip('\n')))
                    file_list.append(fn_list)
                    self.seq_len.append(len(fn_list))
                    self.fns.append(self.root + fn)
        self.trainaug = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.CenterCrop(self.crop_size, self.crop_size)
        ], p=1)

        self.trainaug = A.Compose([
            A.Resize(self.image_size, self.image_size, p=1),
            A.ShiftScaleRotate(rotate_limit=15, p=0.5),
            A.RandomCrop(self.crop_size, self.crop_size, p=1),
            A.RandomBrightnessContrast(p=0.5),
            A.Blur(p=0.5),
            # A.OpticalDistortion(p=0.5), # TODO
        ])

        self.testaug = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.CenterCrop(self.crop_size, self.crop_size)
            ], p=1)

        self.n_images = len(self.image_list)
        if local_rank==0:
            print('The # of videos is:', self.n_images, 'on', self.mode, 'mode!')



    def transform_train(self, img_list, npfile, len1, inner_transform=None, fn=None):
        data, label = {}, {}

        imgs = []

        x = []

        for ind, ims in enumerate(img_list):
            assert os.path.isfile(ims)
            im = cv2.imread(ims)
            if im is None:
                continue
            if ind > 0:
                im = cv2.resize(im, dsize=(col, row))
            else:
                row, col, c = im.shape
            x.append(im[:,:,:,None])
            
        len1 = len(x)
        torch.save(x, 'ct_test.pth')

        # Make the ct scan has sufficient slices
        ########  샘플링 수정 ################## - 32 보다 작으면 FREQ = 1 그리고 16보다 작으면 x2[:-1] = x[-1]
        if len1 < self.FRR*self.FREQ:
            if len1 < 16 :
                x2 = np.zeros((16, x[0].shape[0], x[0].shape[1], x[0].shape[2], x[0].shape[3]), np.uint8)
                print('make the ct scan has sufficient slices', x2.shape)                   
                
#                 print(len(x))
#                 sys.exit()
                
                x2[:len(x)] = x
                x2[len(x):] = x[-1]
                x = x2
                
                self.FREQ = 1
              #  assert len(x) == self.FRR*self.FREQ
                
            else: 
                self.FREQ = 1
               # assert len(x) == self.FRR*self.FREQ
            len1 = len(x)
            x = np.concatenate(x, axis=-1)[:,:,0,:] # (224, 224, 200) ###????
            
            # 전체 16 슬라이스에서 FREQ 1로 설정하고 뽑는 방법
            nStep = self.FREQ #int(np.floor((len1-1) / (self.FRR-1)))
           # max_ind = nStep * (self.FRR-1) # 48
           # max_ind = len1 if max_ind>=len1-1 else max_ind #45
            max_ind = 15

            max_init_ind = len1 - self.FRR*self.FREQ # 200-32 = 168
            init_ind = random.randint(0, max_init_ind)
            print("max_ind:",max_ind)
            print("max_init_ind:",max_init_ind)
            print("init_ind:",init_ind)
            print("len:",len1)
            ind = list(range(init_ind, max_ind+init_ind+1, nStep)) # 0 45
            print("ind:",ind)
            if len(ind)<self.FRR:
                ind = range(self.FRR)
            x = x[:,:,ind]  # (512, 512, 16)    
           # x = [x[index] for index in ind ]    

        else:
            x = np.concatenate(x, axis=-1)[:,:,0,:] # (224, 224, 200) ###????
            
            
            # 전체 슬라이스에서 FREQ 무시하고 z-axis 16으로 rescale해서 뽑는 방법
            nStep = int(np.floor((len1-1) / (self.FRR-1)))
            max_ind = nStep * (self.FRR-1) # 48
            max_ind = len1 if max_ind>=len1-1 else max_ind #45
            
            max_init_ind = len1 - (self.FRR-1)*nStep# 200-32 = 168
            if max_init_ind < 0:
                max_init_ind = len1 - self.FRR*(nStep-1)
           # print('max_init_ind2:',max_init_ind)     
            init_ind = random.randint(0, max_init_ind-1)
#             print('init_ind:',init_ind)
#             print('max_ind:',max_ind)
            
            ind = list(range(init_ind, max_ind+init_ind+1, nStep)) # 0 45
            if len(ind)<self.FRR:
                ind = range(self.FRR)
           # x = [x[index] for index in ind ]
            x = x[:,:,ind]  # (512, 512, 16)
            
            
         ########  샘플링 수정 ##################
         ########  샘플링 수정 전 ##################
#         if len1 < self.FRR*self.FREQ:
#             x2 = np.zeros((self.FRR*self.FREQ, x[0].shape[0], x[0].shape[1], x[0].shape[2], x[0].shape[3]), np.uint8)
#             print('make the ct scan has sufficient slices', x2.shape)
#             x2[:len(x)] = x
#             x2[len(x):] = x[-1]
#             x = x2
#             assert len(x) == self.FRR*self.FREQ
            
            

#         # Random sampling along z-axis
#         len1 = len(x)
#         x = np.concatenate(x, axis=-1)[:,:,0,:] # (224, 224, 200)
#         max_init_ind = len1 - self.FRR*self.FREQ # 200-32 = 168
#         init_ind = random.randint(0, max_init_ind)
#         ind = list(range(init_ind, init_ind+self.FRR*self.FREQ, self.FREQ))  # 16
#         x = x[:,:,ind]  # (512, 512, 16)
        ########  샘플링 수정 전 ##################
    

        # 3D volume augemntation
        x = self.trainaug(image=x)  # (224, 224, 16)

        # Convert to Torch tensor
        x = x['image'].transpose((2, 0, 1))  # (16, 224, 224)
        x = x[:,None,:,:].repeat(3,axis=1)  # (16, 1, 224, 224) -> (16, 3, 224, 224)
        

        x = (np.array(x) - 127.5) / 128.0 # Normalize sholud be done after augmentations
        x = x.astype('float32')

        x = torch.Tensor(x)
        data['img'] = {}
        data['fn'] = fn
        for ind, item in enumerate(x):
            data['img'][ind] = item

        return data

    def transform_test(self, img_list, npfile, len1, inner_transform=None, fn=None):
#         data, label = {}, {}

#         imgs = []


#         x = []
#         # Load volume with the same shape
#         for ind, ims in enumerate(img_list):
#             assert os.path.isfile(ims)
#             im = cv2.imread(ims)
#             if im is None:
#                 continue
#             if ind>0:
#                 im = cv2.resize(im, dsize=(col, row))
#             else:
#                 row, col, c=im.shape
#             x.append(im)

#         len1 = len(x)

#         # Make the ct scan has sufficient slices
#         if len1<self.FRR*self.FREQ:
#             x2 = np.zeros((self.FRR*self.FREQ, x[0].shape[0], x[0].shape[1], x[0].shape[2]), np.uint8)
#             print('make the ct scan has sufficient slices', x2.shape)
#             x2[:len(x)] = x
#             x2[len(x):] = x[-1]
#             x=x2
#             len1 = len(x)

#             assert len(x)==self.FRR*self.FREQ

#         FREQ = self.FREQ
#         if self.test_aug==0: # 전체 슬라이스에서 FREQ 무시하고 z-axis 16으로 rescale해서 뽑는 방법

#             nStep = int(np.floor((len1-1) / (self.FRR-1)))
#             max_ind = nStep * (self.FRR-1) # 48
#             max_ind = len1 if max_ind>=len1-1 else max_ind #45
#             ind = list(range(0, max_ind+1, nStep)) # 0 45
#             if len(ind)<self.FRR:
#                 ind = range(self.FRR)
#             x = [x[index] for index in ind ]

#         elif self.test_aug==1:  # K-crops data augmentation in testing phase
#             # Center에서 인접한 16개 10개 뽑는 것 -> 무조건 inference
#             max_det=self.max_det # 10
#             assert len1>=self.FRR
#             while len1< self.FRR*FREQ: # len1 < 32
#                 FREQ-=1
#                 if FREQ==1:
#                     break

#             targetLen = self.FRR*FREQ + max_det*FREQ # 16*2 + 10*2 = 52
#             if len1 > targetLen: # if len = 80
#                 rem = len1 - targetLen # rem = 28
#                 rem = int(np.floor(rem/2)) # rem = 14
#                 x = [x[ind] for ind in range(rem,rem+targetLen)] # 14 ~ 66
#             else:
#                 x = [x[ind] for ind in range(len1)]

#             len1 = len(x)
#         # test 할때 random으로
#         elif self.test_aug==2:  # center-crops data augmentation in testing phase
#             if self.centerCrop>0 and len1 > self.FRR / self.centerCrop:
#                 x = self.cc(x) # 그냥 가운데 기준 16개 뽑는 것

#             len1 = len(x)
        
        ##### 수정 ###### ##### 수정 ###### ##### 수정 ###### ##### 수정 ###### ##### 수정 ######
        
        data, label = {}, {}

        imgs = []

        x = []

        for ind, ims in enumerate(img_list):
            assert os.path.isfile(ims)
            im = cv2.imread(ims)
            if im is None:
                continue
            if ind > 0:
                im = cv2.resize(im, dsize=(col, row))
            else:
                row, col, c = im.shape
            x.append(im[:,:,:,None])
            
        len1 = len(x)

        # Make the ct scan has sufficient slices
        ########  샘플링 수정 ################## - 32 보다 작으면 FREQ = 1 그리고 16보다 작으면 x2[:-1] = x[-1]
        if len1 < self.FRR*self.FREQ:
            if len1 < 16 :
                x2 = np.zeros((16, x[0].shape[0], x[0].shape[1], x[0].shape[2], x[0].shape[3]), np.uint8)
                print('make the ct scan has sufficient slices', x2.shape)                   
                
#                 print(len(x))
#                 sys.exit()
                
                x2[:len(x)] = x
                x2[len(x):] = x[-1]
                x = x2
                
                self.FREQ = 1
              #  assert len(x) == self.FRR*self.FREQ
                
            else: 
                self.FREQ = 1
               # assert len(x) == self.FRR*self.FREQ
            len1 = len(x)
            x = np.concatenate(x, axis=-1)[:,:,0,:] # (224, 224, 200) ###????
            
            # 전체 16 슬라이스에서 FREQ 1로 설정하고 뽑는 방법
            nStep = self.FREQ  #int(np.floor((len1-1) / (self.FRR-1)))
           # max_ind = nStep * (self.FRR-1) # 48
           # max_ind = len1 if max_ind>=len1-1 else max_ind #45
            max_ind = 15
            max_init_ind = len1 - self.FRR*self.FREQ # 200-32 = 168
            #init_ind = random.randint(0, max_init_ind)
            init_ind = 0 
            ind = list(range(init_ind, max_ind+init_ind+1, nStep)) # 0 45
            if len(ind)<self.FRR:
                ind = range(self.FRR)
            
            x = x[:,:,ind]  # (512, 512, 16)  
            x = inner_transform(image=x)['image']
           # x = [x[index] for index in ind ]    

        else:
            x = np.concatenate(x, axis=-1)[:,:,0,:] # (224, 224, 200) ###????
            
            
            # 전체 슬라이스에서 FREQ 무시하고 z-axis 16으로 rescale해서 뽑는 방법
            nStep = int(np.floor((len1-1) / (self.FRR-1)))
            max_ind = nStep * (self.FRR-1) # 48
            max_ind = len1 if max_ind>=len1-1 else max_ind #45
            
            max_init_ind = len1 - (self.FRR-1)*nStep# 200-32 = 168
            if max_init_ind < 0:
                max_init_ind = len1 - self.FRR*(nStep-1) 
           # init_ind = random.randint(0, max_init_ind-1)
            init_ind = 0 
#             print('init_ind:',init_ind)
#             print('max_ind:',max_ind)
            
            ind = list(range(init_ind, max_ind+init_ind+1, nStep)) # 0 45
            if len(ind)<self.FRR:
                ind = range(self.FRR)
           # x = [x[index] for index in ind ]
            x = x[:,:,ind]  # (512, 512, 16)   
            x = inner_transform(image=x)['image']
        ##### 수정 ###### ##### 수정 ###### ##### 수정 ###### ##### 수정 ###### ##### 수정 ######
        ## Center crop


        # Convert to Torch tensor
        x = x.transpose((2, 0, 1))  # (16, 224, 224)
        x = x[:,None,:,:].repeat(3,axis=1)  # (16, 1, 224, 224) -> (16, 3, 224, 224)
        

        x = (np.array(x) - 127.5) / 128.0 # Normalize sholud be done after augmentations
        x = x.astype('float32')

        x = torch.Tensor(x)
#         print(x.shape)
        
#         x2 = []
#         for im in x:
#             x2.append(inner_transform(image=im)['image'])
#         x = (np.array(x2) - 127.5)/128.0        
#         x = torch.Tensor(x).permute(0, 3, 1, 2)
        data['img']={}
        data['fn'] =fn
        for ind, item in enumerate(x):
            data['img'][ind] = item

        return data

    def __len__(self):
        return self.n_images

    # def randRotate(self, x):
    #     if rn.random()>=0.5:
    #         x = x[:, ::-1, :, :].copy()
    #     if rn.random()>=0.5:
    #         x = x[:, :, ::-1, :].copy()
    #     return x
    #
    # def cc(self, x):
    #     len1 = len(x)
    #     FREQ = self.FREQ
    #     perc = self.centerCrop  # perc이 volume에서 슬라이스 뽑는 시작적ㅁ을 정해줌
    #     targetLen = int(len1 * perc)
    #     while(self.FRR > targetLen):
    #         perc += 0.1
    #         targetLen = int(len1 * perc)
    #         if perc>1:
    #             raise
    #             break
    #
    #
    #     if len1 > targetLen:
    #         rem = len1 - targetLen
    #         rem = int(np.floor(rem/2))
    #         x = [x[ind] for ind in range(rem,rem+targetLen)]
    #
    #     return x

    def __getitem__(self, index):
        img_list = self.image_list[index]
        len1 = self.seq_len[index]
        fn = self.fns[index]
        #(745, 512, 512)

        if self.mode == "test":
            im = self.transform_test(img_list, fn, len1, inner_transform=self.testaug, fn=fn) # (16, 256, 256)
        else:
            im = self.transform_train(img_list, fn, len1, inner_transform=self.testaug, fn=fn)  # (16, 256, 256)
        lab = self.label[index]

        if self.evalPerformance or self.mode=='train':
            return im, lab
        else:
            return im


    
