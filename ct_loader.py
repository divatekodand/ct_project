from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import pdb
import random
import pydicom
import scipy
import scipy.ndimage
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from skimage import exposure
import pickle

glabels2d = None
glabels3d = None
gct3d_desc = None


def is_dcmdir(directory, count_thresh=150):
    files = os.listdir(directory)
    num_dcm = len([1 for file in files if file[-4:] == '.dcm'])
#     print('len - ', num_dcm)
    if num_dcm > count_thresh:
        return True
    else:
        return False


class CTLoader(data.Dataset):
    """
    Dataloader for CT Image Analysis
    """
    def __init__(self,
               root_dir_2d,
               label_file_2d,
               root_dir_3d,
               label_file_3d=None,
               num_classes=6,
               groups=128,
               split="train",
               num_val=1000,
               sample = 0,
               label_file_2d_sample='',
               transforms=None,
               preprocessed2d=None,
               aug2d = True,
               oversample=True):
        assert split in ["train", "val", "test"]
        
        global glabels2d, glabels3d, gct3d_desc
        # root folder, split

        self.root_dir_2d = os.path.abspath(root_dir_2d)
        self.label_file_2d = os.path.abspath(label_file_2d)
        self.label_file_2d_sample = label_file_2d_sample
        self.sample = sample
        self.labels_2d_matfile = os.path.abspath('./labels_2d.mat')
        self.preprocessed2d = preprocessed2d
        self.aug2d = aug2d

        self.root_dir_3d = os.path.abspath(root_dir_3d)
        self.label_file_3d = os.path.abspath(label_file_3d)
        self.labels_3d_matfile = os.path.abspath('./labels_3d.mat')

        self.split = split
        self.transforms = transforms
        self.n_classes = num_classes
        self.groups = groups
        self.scantype = {'2d': '2d', '3d': '3d'}
        self.num_val = num_val

        # load all labels
        if self.preprocessed2d is not None:
            files = os.listdir(self.preprocessed2d)
            self.label_file_2d = [os.path.join(self.preprocessed2d, f) for f in files if os.path.exists(os.path.join(self.preprocessed2d, f))]
            glabels2d = self.label_file_2d
            self.num_files_2d = len(files)
            if self.split == 'train':
                self.num_2d = self.num_files_2d - self.num_val
                self.offset = 0
            else:
                self.num_2d = self.num_val
                self.offset = self.num_files_2d - self.num_val
        else:
            if not os.path.exists(self.label_file_2d):
              raise ValueError(
                'Label file {:s} does not exist!'.format(self.label_file_2d))
            print('Getting 2d labels - ')
            if glabels2d is not None:
                self.labels_2d = glabels2d
            else:
                self.labels_2d = self.get_labels_2d()
                glabels2d = self.labels_2d

            self.num_2d = len(self.labels_2d) // groups
            if split == 'train':
                self.offset = 0
                self.num_2d = self.num_2d - self.num_val
            else:
                self.offset = self.num_2d - self.num_val
                self.num_2d = self.num_val

        if not os.path.exists(self.label_file_3d):
          raise ValueError(
            'Label file {:s} does not exist!'.format(self.label_file_3d))
        print('Getting 3d labels - ')
        if glabels3d is not None:
            self.labels_3d = glabels3d
        else:
            self.labels_3d = self.get_labels_3d()
            glabels3d = self.labels_3d

        self.num_3d = len(self.labels_3d)

        if gct3d_desc is not None:
            self.ct3d_desc = gct3d_desc
        else:
            cache = './cache/ct3d_desc.pkl'
            if os.path.exists(cache):
                with open(cache, 'rb') as f:
                    self.ct3d_desc = pickle.load(f)
            else:
                self.ct3d_desc = self.ct3d_descriptor()
                gct3d_desc = self.ct3d_desc 
                if not os.path.exists('./cache'):
                    os.mkdir('./cache')
                with open(cache, 'wb') as f:
                    pickle.dump(self.ct3d_desc, f)

        ct3d_desc_filtered_keys = self.ct3d_desc['tree'].keys()
        filtered_labels = []
        for case, label in self.labels_3d:
            if case in ct3d_desc_filtered_keys:
                filtered_labels.append((case, label))
        
        self.labels_3d = filtered_labels
        self.num_3d = len(self.labels_3d)
        glabels3d = self.labels_3d

        # Oversampling
        self.oversample = oversample
        self.oversample_flag = False

        if split == "train" and self.oversample:
            stats, indices = self.get_oversample_stats()
            self.oversample_stats = stats
            self.oversample_indices = indices
            self.num_2d = len(self.oversample_indices)


    
    def __len__(self):
        n_samples = self.num_2d #+ self.num_3d
        return n_samples

    def __getitem__(self, index):
        """
        input tensor shape - n, 128, 200,200
        output tensor shape - n, 128, 6
        """
        scans_list = []
        labels_list = []
        filenames = []
        if index < self.num_2d:
            #return #groups 2d samples
            #print('index - ',  index)
            if self.preprocessed2d:
                index = index + self.offset
                if self.split =="train" and self.oversample and self.oversample_flag:
                    index = self.oversample_idx(index)
                # print(self.split, index)
                batch_dict = sio.loadmat(self.label_file_2d[index])
                scans = batch_dict['x']
                labels = batch_dict['y']
                if self.aug2d:
                    scans, labels = self.augment_2d(scans, labels)
                filenames = batch_dict['filenames']
            else:
                start_index = (self.offset * self.groups) + index * self.groups
                end_index = (self.offset * self.groups) + (index + 1) * self.groups
                scan_type = self.scantype['2d']
                for i in range(start_index, end_index):
                    filename, label = self.labels_2d.index.values[i], self.labels_2d.iloc[i,0:len(self.labels_2d.columns)].tolist()
                    filenames.append(filename)
                    filename = filename + '.dcm'
                    #filename = 'ID_0000aee4b.dcm'
                    ds = pydicom.dcmread(os.path.join(self.root_dir_2d, filename))
                    scans_list.append(ds)
                    labels_list.append(label)
                    scans, labels = self.preprocess(scans_list, labels_list, scan_type)
                    sizes = [s.pixel_array.shape for s in scans_list]
                    #print('2d sizes - ', sizes)
        else:
            # return 3d sample
            scan_type = self.scantype['3d']
            index3d = index - self.num_2d
            #pdb.set_trace()
            example_key, label = self.labels_3d[index3d]
            leaf_dirs = self.ct3d_desc['tree'][example_key]
            #leaf_dirs = self.ct3d_desc['tree']['CQ500-CT-0']
            dcm_dir = random.choice(leaf_dirs)
            filenames.append(dcm_dir)
            file_names = os.listdir(dcm_dir)
            dcm_names = sorted([ file_name for file_name in file_names if file_name[-3:]=='dcm'])
            file_names = sorted([ file_name for file_name in file_names if file_name[-3:]=='mat'])
            #print('num dcm - ', len(dcm_names))
            #print('num mat - ', len(file_names))
            ds = pydicom.dcmread(os.path.join(dcm_dir, dcm_names[0]))
            spacing = ds.PixelSpacing
            slope = ds.RescaleSlope
            slice_thickness = ds.SliceThickness
            intercept = ds.RescaleIntercept
            labels_list = label
            for files in zip(file_names, dcm_names):
                ds = sio.loadmat(os.path.join(dcm_dir, files[0]))
                meta = pydicom.dcmread(os.path.join(dcm_dir, files[1]))
                scans_list.append((ds['frame'], int(meta.InstanceNumber)))
            scans_list.sort(key = lambda x: int(x[1]))
            scans_list = [scan[0] for scan in scans_list]
            scans, labels = self.preprocess(scans_list, labels_list, scan_type, intercept, slope, spacing, slice_thickness)
        return scans, labels #, filenames

    
    def augment_2d(self, scans, labels):
        nchannels = scans.shape[0]
        assert nchannels == labels.shape[0], "2d Samples tensor shape does not match that of the labels"
        
        # shuffle indicies
        index = np.array(list(range(nchannels)))
        np.random.shuffle(index)
        scans = scans[index, :, :]
        labels = labels[index]

        # Add random noise
        if random.choice([True, False]):
            epsilon = 0.0001
        else:
            epsilon = 0.001
        
        mask = np.random.randint(-1, 2, size=scans.shape)
        scans += epsilon * mask

        return scans, labels

    def crop_centre(self, image, out_shape=[200, 200]):
        out_shape = [self.groups] + out_shape
        image_shape = image.shape
        startx = image_shape[0]//2-(out_shape[0]//2)
        starty = image_shape[1]//2-(out_shape[1]//2)
        startz = image_shape[2]//2-(out_shape[2]//2)
        return image[startx:startx+out_shape[0], starty:starty+out_shape[1],startz:startz+out_shape[2]]

    def normalize(self, image, min_bound=-1000.0, max_bound=400.0, mean=0.25):
        image = (image - min_bound) / (max_bound - min_bound)
        image[image>1] = 1
        image[image<0] = 0
        image -= mean
        return image

    def preprocess(self, scans_list, labels_list, scan_type, intercept=None, slope=None, spacing=None, slice_thickness=None ,new_spacing = [1,1,1]):
        labels = torch.tensor(labels_list, dtype=torch.float32)
        if scan_type == '2d':
            intercept = scans_list[0].RescaleIntercept
            #print('intercept - ', intercept)
            slope = scans_list[0].RescaleSlope
            #print('slope - ', slope)
            spacing = scans_list[0].PixelSpacing
            image = np.stack([s.pixel_array for s in scans_list])
            # Convert to int16 (from sometimes int16),
            # should be possible as values should always be low enough (<32k)
            image = image.astype(np.int16)
        else:
            image = np.stack(scans_list)
            image = image.astype(np.int16)
        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0
        # Convert to Hounsfield units (HU)
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        image = np.array(image, dtype=np.int16)
        # resample
        new_spacing = np.array(new_spacing)
        if scan_type == '2d':
            spacing = np.array([1] + list(map(float, spacing)))
        else:
            spacing = [float(slice_thickness)] + list(map(float, (spacing)))
            spacing = np.array(spacing)
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        image = self.normalize(image)
        image = self.crop_centre(image)
        image = torch.tensor(image, dtype=torch.float32)
        return image, labels

    def oversample_idx(self, i):
        return self.oversample_indices[i]

    def get_oversample_stats(self, over_factor=None, cache="./cache/oversample_stats.pth.tar"):
        """
        Any ~ 14%
        Epidural ~ 0.5%
        Each of the other ~ 4%
        Oversampling Heuristics for each set of 128 2d slices -> epidural_factor * (is_epidural) + any_factor
        over_factor = {"any":2, "epidural": 8} # default
        stats - dictionary from index to oversampling factor
        """

        if over_factor is None:
            over_factor = {"any":2, "epidural": 8}

        if os.path.exists(cache):
            stats = torch.load(cache)
            indices = stats[-1]    
        else:
            stats = {}
            # stats_tensors = {}

            for i in range(self.__len__()):
                print("oversample stats prog - ", i, " / ", self.__len__())
                scans, labels = self.__getitem__(i)
                freqs = np.sum(labels, axis=1)
                is_any = 1 if freqs[0]>0 else 0
                is_epidural = 1 if freqs[1]>0 else 0
                factor = over_factor["any"] * is_any + over_factor["epidural"] * is_epidural
                stats[i] = (self.label_file_2d[i], factor)
                # stats_tensors[i] = (self.label_file_2d[i], labels)

            indices = []
            for key in stats:
                for i in range(stats[key][1]):
                    indices.append(key)

            stats[-1] = indices
            torch.save(stats, cache)
        
        self.oversample_flag = True

        return stats, indices

    def get_labels_2d_helper(self):
        labels_2d = pd.read_csv(self.label_file_2d)
        labels_2d.dropna(inplace = True)
        new = labels_2d['ID'].str.split("_", n=2, expand=True)
        labels_2d["Case_ID"] = new[0]+ '_' +new[1]
        labels_2d["Type"] = new[2]
        labels_2d.drop(columns = ["ID"], inplace = True)
        labels_2d.rename(columns={'Case_ID':'ID'}, inplace=True)
        columnsTitles=["ID","Type","Label"]
        labels_2d = labels_2d.reindex(columns=columnsTitles)
        labels_2d.drop_duplicates(keep=False,inplace=True)
        labels_2d = labels_2d.pivot(index='ID', columns='Type', values='Label')
        return labels_2d

    def get_labels_2d(self):
        if self.sample > 0:
            if os.path.exists(self.label_file_2d_sample):
                labels_2d = pd.read_csv(self.label_file_2d_sample)
            else:
                labels_2d = self.get_labels_2d_helper()
                labels_2d = labels_2d.head(self.sample)
                labels_2d.to_csv(self.label_file_2d_sample, index=True)
        else:
            if os.path.exists(self.labels_2d_matfile):
                labels_2d = pd.read_csv(self.labels_2d_matfile)
            else:
                labels_2d = self.get_labels_2d_helper()
                labels_2d.to_csv(self.labels_2d_matfile, index=True)
        labels = labels_2d
        #labels = []
        
        #print('get labels 2d for loop')
        #for i in range(len(labels_2d)):
        #    filename, label = labels_2d.iloc[i,0], labels_2d.iloc[i,1:len(labels_2d.columns)].tolist()
        #    labels.append((filename, label))
        return labels

    def get_labels_3d(self):
        labels_3d = pd.read_csv(self.label_file_3d)
        labels_3d['any'] = (labels_3d['R1:ICH'] + labels_3d['R2:ICH'] + labels_3d['R3:ICH']) // 2
        labels_3d['epidural'] = (labels_3d['R1:EDH'] + labels_3d['R2:EDH'] + labels_3d['R3:EDH']) // 2
        labels_3d['intraparenchymal'] = (labels_3d['R1:IPH'] + labels_3d['R2:IPH'] + labels_3d['R3:IPH']) // 2
        labels_3d['intraventricular'] = (labels_3d['R1:IVH'] + labels_3d['R2:IVH'] + labels_3d['R3:IVH']) // 2
        labels_3d['subarachnoid'] = (labels_3d['R1:SAH'] + labels_3d['R2:SAH'] + labels_3d['R3:SAH']) // 2
        labels_3d['subdural'] = (labels_3d['R1:SDH'] + labels_3d['R2:SDH'] + labels_3d['R3:SDH']) // 2
        classes = self.get_index_mapping()
        cols = labels_3d.columns.tolist()
        cols = [col for col in cols if col not in classes]
        cols.remove('name')
        labels_3d.drop(cols, axis=1, inplace=True)
        labels = []
        for i in range(len(labels_3d)):
            filename, label = labels_3d.iloc[i,0], labels_3d.iloc[i,1:len(labels_3d.columns)].tolist()
            labels.append((filename, label))
        return labels

    def ct3d_descriptor(self):
        cq500_dataset = {}
        cq500_dataset['root_dir'] = self.root_dir_3d
        cq500_dataset['csv_path'] = os.path.join(cq500_dataset['root_dir'],'reads.csv')
        #TODO: Sorted output not in order because of different number of characters in file names.
        cq500_dataset['case_list'] = sorted(os.listdir(cq500_dataset['root_dir']))
        cq500_dataset['case_list'] = [os.path.join(cq500_dataset['root_dir'], case) for case in cq500_dataset['case_list'] if case !=os.path.basename(cq500_dataset['csv_path'])]
        # print(cq500_dataset['case_list'])
        print('Getting Directory Structure for each 3D CT folder -')
        #print_tree(cq500_dataset['case_list'][0])
        cq500_dataset['tree'] = self.get_filtered_ctdirs(cq500_dataset)
        return cq500_dataset

    def get_filtered_ctdirs(self, cq500_dataset):
        #Create a dictionary containing all the valid path (to feed this into the dataloader).
        #Key - root_dir_path (string),
        #Value - list of paths (each path is a valid dcm folder for the same case)
        cq500_dict = {}
        for case_dir in cq500_dataset['case_list']:
            for root, dirs, files in os.walk(case_dir):
        #             print('root - ', root)
                if is_dcmdir(root):
                    #print(root)
                    root_split = root.split(os.sep)
                    #print('root split - ', root_split)
                    #case = '/'.join(root_split[1:6])
                    index = 0
                    for i in range(len(root_split)):
                        if 'CQ500' in root_split[i]:
                            index = i
                            break
                    case = root_split[index]
                    #mode = '/'.join(root_split[6:])
                    if case in cq500_dict:
                        cq500_dict[case].append(root)
                    else:
                        cq500_dict[case] = [root]
        return cq500_dict

    def get_index_mapping(self):
        # return a dict from labels to category name
        labels = [ 'any',  'epidural',  'intraparenchymal',  'intraventricular',  'subarachnoid', 'subdural']
        return labels


class CTLoader_dummy(data.Dataset):
    """
    Dataloader for CT Image Analysis
    """

    def __init__(self,
                 root_dir_2d=None,
                 label_file_2d=None,
                 root_dir_3d=None,
                 label_file_3d=None,
                 num_classes=6,
                 groups=1,
                 split="train",
                 num_val=1,
                 num_samples=1,
                 sample=0,
                 label_file_2d_sample='',
                 modalities=1,
                 transforms=None):
        assert split in ["train", "val", "test"]
        self.n_classes = num_classes
        self.groups = groups
        self.num_val = num_val
        self.num_samples = num_samples
        self.modalities = modalities

    def __len__(self):

        return self.num_samples

    def __getitem__(self, index):
        # load img and label
        dim = random.choice([0, 1])
        if self.modalities == 1:
            dim = 0
        if dim == 0:
            # return torch.rand((self.groups,200,200), dtype=torch.float32), torch.ones((6*self.groups), dtype=torch.float32)
            return 0.1*torch.rand((self.groups, 100, 100), dtype=torch.float32), (
                        torch.rand((self.groups, self.n_classes), dtype=torch.float32) > 0.5).float()
        else:
            # return torch.rand((self.groups,200,200), dtype=torch.float32), torch.ones((6), dtype=torch.float32)
            return 0.1*torch.rand((self.groups, 100, 100), dtype=torch.float32), (
                        torch.rand((self.groups, self.n_classes), dtype=torch.float32) > 0.5).float()


def histogram_equalize(img):
   img_cdf, bin_centers = exposure.cumulative_distribution(img)
   return np.interp(img, bin_centers, img_cdf)


def main_dummy():
    train_dataset = CTLoader_dummy()
    val_dataset = CTLoader_dummy()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, sampler=None, drop_last=False)

    print("Train Loader Tensor shapes")
    for i, (x, y) in enumerate(train_loader):
        print(x.shape)
        print(y.shape)
        if i > 10:
            break

    print("Val Loader Tensor shapes")
    for i, (x, y) in enumerate(val_loader):
        print(x.shape)
        print(y.shape)
        if i > 10:
            break


def main():
    train_dataset = CTLoader(args.root_dir_2d, args.label_file_2d, args.root_dir_3d, args.label_file_3d,
                                   groups=args.group, split='train', num_val=args.num_val, sample=True)
    val_dataset = CTLoader(args.root_dir_2d, args.label_file_2d, args.root_dir_3d, args.label_file_3d,
                                 groups=args.group, split='val', num_val=args.num_val, sample=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    print("Train Loader Tensor shapes")
    for i, (x, y) in enumerate(train_loader):
        print(x.shape)
        print(y.shape)
        if i > 10:
            break

    print("Val Loader Tensor shapes")
    for i, (x, y) in enumerate(val_loader):
        print(x.shape)
        print(y.shape)
        if i > 10:
            break

    a = train_dataset[601]
    print('train dataset shape - ', a[0].shape)
    plt.hist(a[0].flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()
    # Show some slice in the middle
    plt.imshow(histogram_equalize(a[0][0, :, :].numpy()), cmap=plt.cm.gray)
    plt.show()

if __name__=='__main__':
    # main()
    main_dummy()



