import os
import random
import socket
import time

import albumentations as augs
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from skimage import transform as trans

host_name = socket.gethostname()

def get_scale_center(bb, scale_=220.0):
    center = np.array([bb[2] - (bb[2] - bb[0]) / 2, bb[3] - (bb[3] - bb[1]) / 2])
    scale = (bb[2] - bb[0] + bb[3] - bb[1]) / scale_
    return scale, center


def inv_mat(mat):
    ans = np.linalg.pinv(np.array(mat).tolist() + [[0, 0, 1]])
    return ans[:2]


def get_transform(center, scale, res, rot=0):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1

    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 200
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))

    return t

def preprocess(img_name, rects,pts, img_name1, pts1, img_name2, pts2, im_w, scale, rng, phase, transform, flip):
    image = cv2.imread(img_name)
    image1 = cv2.imread(img_name1)
    image2 = cv2.imread(img_name2)
    lmk = np.array([float(x) for x in pts.split(',')], dtype=np.float32)
    lmk = lmk.reshape((5, 2))
    lmk1 = np.array([float(x) for x in pts1.split(',')], dtype=np.float32)
    lmk1 = lmk1.reshape((5, 2))
    lmk2 = np.array([float(x) for x in pts2.split(',')], dtype=np.float32)
    lmk2 = lmk2.reshape((5, 2))

    bb = np.array([float(x) for x in rects.split(',')], dtype=np.float32)
    assert scale>0
    scale, center = get_scale_center(bb, scale_=scale)

    if phase == 'test' and flip:
        flip_flag = True
    else:
        flip_flag = False

    if phase == 'train':
        # shared by 3 images
        aug_rot = (rng.rand() * 2 - 1) * 15  # in deg.
        aug_scale = rng.rand() * 0.25 * 2 + (1 - 0.25)  # ex: random_scaling is .25
        scale *= aug_scale
        scale = max(scale, 0.1)
        dx = rng.randint(-10 * scale, 10 * scale) / center[0]  # in px
        dy = rng.randint(-10 * scale, 10 * scale) / center[1]
        if rng.rand() > 0.5:
            flip_flag = True

    else:
        aug_rot = 0
        dx, dy = 0,0

    center[0] += dx * center[0]
    center[1] += dy * center[1]
    mat = get_transform(center, scale, (im_w, im_w), aug_rot)[:2]
    img = cv2.warpAffine(image.copy(), mat, (im_w, im_w))

    lmk = np.dot(np.concatenate((lmk, lmk[:, 0:1]*0+1), axis=1), mat.T)

    # align image1 to lmk
    tform = trans.SimilarityTransform()
    tform.estimate(lmk1, lmk)
    M = tform.params[0:2, :]
    image1 = cv2.warpAffine(image1.copy(), M, (im_w, im_w), borderValue=0.0)

    # align image2 to lmk
    tform = trans.SimilarityTransform()
    tform.estimate(lmk2, lmk)
    M = tform.params[0:2, :]
    image2 = cv2.warpAffine(image2.copy(), M, (im_w, im_w), borderValue=0.0)



    if phase == 'train':
        content_transform = augs.Compose([augs.Blur(p=0.5),
                                          augs.ColorJitter(0.3, 0.3, 0.3, 0, 3, p=1.0),
                                          augs.ImageCompression(quality_lower=30, p=0.5),
                                          augs.CoarseDropout(min_holes=1, max_holes=8,
                                                             min_width=0.03125, min_height=0.03125,
                                                             max_width=0.125, max_height=0.125, p=0.5)],
                                         additional_targets={'image1': 'image', 'image2': 'image'})
        img_all = content_transform(image=img, image1=image1, image2=image2)
        img = img_all['image']
        image1 = img_all['image1']
        image2 = img_all['image2']


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(img1)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(img2)

    if flip_flag:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

    img = transform(img)
    img1 = transform(img1)
    img2 = transform(img2)
    return img, img1, img2

class data_load(data.Dataset):
    def __init__(self, data='bp4d', phase='train', subset=None, flip=False,transform=None,seed=0):
        self.rng = np.random.RandomState(seed=seed)
        self.flip = flip
        self.phase = phase
        self.scale = 260
        self.transform = transform
        self.interv = 10
        self.data = data

        if data=='bp4d':
            predix_path = '/data/home/jinang/cluster/Datasets/BP4D/'  # annotation path
            self.predix_path = predix_path + '/BP4D-small/'  # image path
            img_list_all = open(predix_path + '/bp4d_small_retina_anno/BP4D_names_label.txt').readlines()
            rect_5pts_list_all = open(predix_path + '/bp4d_small_retina_anno/BP4D_rect_5pts_label.txt').readlines()
            label_list_all = np.loadtxt(predix_path + '/bp4d_small_retina_anno/au_labels.txt')
            if self.phase == 'train':
                paths = open('data/BP4D/splits/BP4D_tr%d_path.txt' % subset).readlines()
            else:
                paths = open('data/BP4D/splits/BP4D_ts%d_path.txt' % subset).readlines()
            self.img_list = []
            self.rect_5pts_list = []
            self.label_list = []
            start_t = time.time()
            for path_ in paths:
                path_ = path_.replace('_', '/')
                if path_ in img_list_all:
                    index = img_list_all.index(path_)
                    self.img_list.append(img_list_all[index])
                    self.rect_5pts_list.append(rect_5pts_list_all[index])
                    self.label_list.append(label_list_all[index])
            end_t = time.time()
            print('take time', end_t - start_t, '%d/%d' % (len(self.label_list), len(paths)))
        
        elif data=='disfa':
            predix_path = '/data/home/jinang/cluster/Datasets/DISFA/'
            self.predix_path = predix_path+'/DISFA_frames/'
            img_list_all = open(predix_path + '/DISFA_retina_anno/DISFA_names_label.txt').readlines()
            rect_5pts_list_all = open(predix_path + '/DISFA_retina_anno/DISFA_rect_5pts_label.txt').readlines()
            label_list_all = np.loadtxt(predix_path + '/DISFA_retina_anno/au_labels.txt')
            if self.phase == 'train':
                paths = []
                full_set = [1, 2, 3]
                full_set.remove(subset)
                for i in full_set:
                    print('train on disfa', i)
                    paths.extend(open('data/DISFA/splits/DISFA_part%d_path.txt' % i).readlines())
            else:
                # print('test on disfa', subset)
                paths = open('data/DISFA/splits/DISFA_part%d_path.txt' % subset).readlines()

            self.img_list = []
            self.label_list = []
            self.rect_5pts_list = []
            start_t = time.time()
            # index in list
            for path_ in paths:
                _, SN, img_name = path_.strip().split('/')
                img_num = int(float(img_name.replace('.jpg', '')))
                path_ = 'LeftVideo' + SN + '_comp_' + '%08d.jpg' % img_num
                index = img_list_all.index(path_ + '\n')
                self.img_list.append(img_list_all[index])
                self.rect_5pts_list.append(rect_5pts_list_all[index])
                self.label_list.append(label_list_all[index])
        else:
            raise NotImplementedError

        self.label_list = np.asarray(self.label_list)
        assert len(self.img_list) == len(self.rect_5pts_list) == self.label_list.shape[0]
        # balance weights
        AUoccur_rate = np.zeros((1, self.label_list.shape[1]))
        for i in range(self.label_list.shape[1]):
            AUoccur_rate[0, i] = sum(self.label_list[:, i] > 0) / float(self.label_list.shape[0])
        AU_weight = 1.0 / AUoccur_rate
        self.AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        im_w = 256
        name = self.img_list[index].strip('\n')
        rects, pts5 = self.rect_5pts_list[index].strip('\n').split(';')
        au = self.label_list[index]
        img_name = self.predix_path + name

        self.interv1 = self.interv
        self.interv2 = self.interv

        if self.data == 'bp4d':
            cur_name = name.split('/')[-1]
            cur_index = int(float(cur_name.replace('.jpg', '')))
            if os.path.isfile(img_name.replace(cur_name, '%d.jpg' % cur_index)):
                former_name = '%d.jpg' % (cur_index - self.interv1)
                later_name = '%d.jpg' % (cur_index + self.interv2)
            elif os.path.isfile(img_name.replace(cur_name, '%02d.jpg' % cur_index)):
                former_name = '%02d.jpg' % (cur_index - self.interv1)
                later_name = '%02d.jpg' % (cur_index + self.interv2)
            elif os.path.isfile(img_name.replace(cur_name, '%03d.jpg' % cur_index)):
                former_name = '%03d.jpg' % (cur_index - self.interv1)
                later_name = '%03d.jpg' % (cur_index + self.interv2)
            elif os.path.isfile(img_name.replace(cur_name, '%04d.jpg' % cur_index)):
                former_name = '%04d.jpg' % (cur_index - self.interv1)
                later_name = '%04d.jpg' % (cur_index + self.interv2)
            elif os.path.isfile(img_name.replace(cur_name, '%05d.jpg' % cur_index)):
                former_name = '%05d.jpg' % (cur_index - self.interv1)
                later_name = '%05d.jpg' % (cur_index + self.interv2)
            else:
                print(img_name)
                print(cur_name)
                print(cur_index)
                raise NotImplementedError()
        elif self.data == 'disfa':
            cur_name = name.split('_')[-1]
            cur_index = int(float(cur_name.replace('.jpg', '')))
            former_name = '%08d.jpg'%(cur_index-self.interv1)
            later_name = '%08d.jpg'%(cur_index+self.interv2)
        else:
            raise NotImplementedError()

        former_img_name = name.replace(cur_name, former_name)
        later_img_name = name.replace(cur_name, later_name)
        index_former = index - self.interv1
        index_later = index + self.interv2

        if not os.path.isfile(self.predix_path + former_img_name):
            former_img_name = name
            index_former = index
        if not os.path.isfile(self.predix_path + later_img_name):
            later_img_name = name
            index_later = index

        if not self.img_list[index_former].strip('\n') == former_img_name:
            # copy current frame
            former_img_name = img_name
            rects_former = rects
            pts_former = pts5
            au_former = au
        else:
            # use previous frame
            former_img_name = self.predix_path + former_img_name
            rects_former, pts_former = self.rect_5pts_list[index_former].strip('\n').split(';')
            au_former = self.label_list[index_former]

        if not self.img_list[index_later].strip('\n') == later_img_name:
            # copy current frame
            later_img_name = img_name
            rects_later = rects
            pts_later = pts5
            au_later = au

        else:
            # use later frame
            later_img_name = self.predix_path + later_img_name
            rects_later, pts_later = self.rect_5pts_list[index_later].strip('\n').split(';')
            au_later = self.label_list[index_later]

        if (former_img_name == img_name) and (later_img_name == img_name):
            flag = 0
        else:
            flag = 1

        img, img_former, img_later = preprocess(img_name, rects, pts5, former_img_name, pts_former, later_img_name,
                                                pts_later, im_w, self.scale, self.rng, self.phase, self.transform,
                                                self.flip)

        label = torch.from_numpy(au.astype('float32'))
        data['img'] = img
        data['img_former'] = img_former
        data['img_later'] = img_later
        data['label'] = label
        data['flag'] = flag
        data['name'] = img_name

        return data


if __name__ == '__main__':
    import torchvision
    for ii in [5, 10, 15, 20, 30]:
        data = bp4d_load(phase='test', subset=1, interv=ii, transform=transforms.Compose([transforms.ToTensor()]))
        data_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        count = 0
        for data in data_loader:
            count = count + data['flag'].sum()
            # torchvision.utils.save_image(data['img'], 'mid.png', normalize=True)
            # torchvision.utils.save_image(data['img_former'], 'former.png', normalize=True)
            # torchvision.utils.save_image(data['img_later'], 'later.png', normalize=True)
            # break
        print(count)
