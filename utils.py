import os


def print_tree(startpath, list_files=False):
    '''
    Prints startpath and tree starting at startpath
    '''
    print('start_path : ', startpath)
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        if list_files:
            for f in files:
                print('{}{}'.format(subindent, f))


# Check for CT scans that are faulty. Typically each CT directory consists of more than 200 dcm files.
# Filter out directories that have very less number of dcm files.
def is_dcmdir(directory, count_thresh=150):
    files = os.listdir(directory)
    num_dcm = len([1 for file in files if file[-4:] == '.dcm'])
    #     print('len - ', num_dcm)
    if num_dcm > count_thresh:
        return True
    else:
        return False


def get_filtered_ctdirs(cq500_dataset):
    # Create a dictionary containing all the valid path (to feed this into the dataloader).
    # Key - root_dir_path (string),
    # Value - list of paths (each path is a valid dcm folder for the same case)
    cq500_dict = {}
    for case_dir in cq500_dataset['case_list']:
        for root, dirs, files in os.walk(case_dir):
            #             print('root - ', root)
            if is_dcmdir(root):
                # root = root.strip()
                root_split = root.split(os.sep)
                # print('root split - ', root_split)
                # case = '/'.join(root_split[1:6])
                index = 0
                for i in range(len(root_split)):
                    if 'CQ500' in root_split[i]:
                        index = i
                        break
                case = root_split[index]
                # mode = '/'.join(root_split[6:])
                if case in cq500_dict:
                    cq500_dict[case].append(root)
                else:
                    cq500_dict[case] = [root]
    return cq500_dict


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.0
    self.avg = 0.0
    self.sum = 0
    self.count = 0.0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def main():
    cq500_dataset = {}
    # cq500_dataset['root_dir'] = '/kaggle/input/cq500-mat/cq500'
    cq500_dataset['root_dir'] = '../data/cq500'
    cq500_dataset['csv_path'] = os.path.join(cq500_dataset['root_dir'], 'reads.csv')
    # TODO: Sorted output not in order because of different number of characters in file names.
    cq500_dataset['case_list'] = sorted(os.listdir(cq500_dataset['root_dir']))
    cq500_dataset['case_list'] = [os.path.join(cq500_dataset['root_dir'], case) for case in cq500_dataset['case_list'] if
                                  case != os.path.basename(cq500_dataset['csv_path'])]
    # print(cq500_dataset['case_list'])
    print('Directory Structure for each 3D CT folder -')
    print_tree(cq500_dataset['case_list'][0])
    cq500_dataset['tree'] = get_filtered_ctdirs(cq500_dataset)

    print('CQ500 Dataset - ', cq500_dataset.keys())

    ## Print number of files in each dcm dir
    # for key in cq500_dataset['tree']:
    #     flist = cq500_dataset['tree'][key]
    #     print(key)
    #     for f in flist:
    #         print(f)
    #         print(len(os.listdir(f)))


if __name__ == '__main__':
    main()
