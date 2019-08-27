import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np
from pdb import set_trace as st

### load n_frame file
def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

### image loader
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

### video loader
def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

### load annotation data
def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

### get class labels
def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

### get video names and annotations
def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
    return video_names, annotations

def get_labels_class(data):
    labels_class_list = []
    for idx, class_label in enumerate(data['labels']):
        labels_class_list.append(idx)
    return labels_class_list

def make_dataset(
                root_path, annotation_path, subset,
                n_samples_for_each_video,
                sample_duration
                ):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = get_labels_class(data)
    dataset, targets = [], []

    frame_index_global = 0
    for i in range( len(video_names) ):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue
        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        ### modify 0814, skip too short video
        if n_frames < sample_duration:
            print("[ERR] {} {}-{}, skipped".format(video_names[i], n_frames, sample_duration))
            continue

        begin_t = 1
        end_t = n_frames # 64
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1],
            'video_index': i,
            'video_index_v': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[ annotations[i]['label'] ]
        else:
            sample['label'] = -1
        if n_samples_for_each_video == 1:
            n_frames = min(n_frames, sample_duration)
            sample['frame_indices_local'] = list(range(1, n_frames + 1))
            sample['frame_indices_global'] = list(range(frame_index_global, n_frames + frame_index_global))
            sample['frame_indices_global2'] = [i*100 + x  for x in range(1, n_frames + 1)]
            frame_index_global += n_frames
            dataset.append(sample)
            targets.append( sample['label'] )
        else: ### modify 0813
            ### case one: vector embedding
            n_frames = min(n_frames, sample_duration)
            width = n_samples_for_each_video # 2 * 4 + 1, overwrite
            for j in range(1, n_frames + 1 - width):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices_local'] = list(range(j, width + j))
                sample_j['frame_indices_global'] = list(range(frame_index_global + j - 1, frame_index_global + width + j - 1 )) ### this one should not be used
                sample_j['frame_indices_global2'] = [i*100 + x  for x in range(j, width + j)]
                sample_j['video_index_v'] = sample_j['video_index'] * (n_frames - width) + j
                sample_j['video_index_v2'] = sample_j['video_index'] * 100 + j
                dataset.append(sample_j)
                targets.append(sample_j['label'])
            frame_index_global += n_frames
    return dataset, targets

class UCF101Instance(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(
                self,
                root_path,
                annotation_path,
                subset,
                transform = None,
                spatial_transform = None,
                temporal_transform = None,
                target_transform = None,
                sample_duration = 16,
                n_samples_for_each_video = 1,
                get_loader = get_default_video_loader
                ):
        self.data, self.targets = make_dataset(
            root_path, annotation_path, subset,
            n_samples_for_each_video, sample_duration
        )
        self.transform = transform
        self.target_transform = target_transform
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices_local'] # index 1 ... n_frames
        frame_indices_global = self.data[index]['frame_indices_global'] # index 1 ... 9700 x n_frames
        frame_indices_global2 = self.data[index]['frame_indices_global2'] # video_index xxx
        video_index = self.data[index]['video_index'] # index 0...9700
        video_index_v = self.data[index]['video_index_v'] # video index v, 0...9700 xxx ++, 2 xxx
        target = self.data[index]['label'] # video_id index 0...101
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3) # like volume
        clip = torch.stack(clip, 0)
        target = torch.tensor([target for i in range(clip.shape[0])], dtype=torch.long)
        video_index = torch.tensor([video_index for i in range(clip.shape[0])], dtype=torch.long)
        video_index_v = torch.tensor([video_index_v for i in range(clip.shape[0])], dtype=torch.long)
        frame_index = torch.tensor(frame_indices_global, dtype=torch.long)
        frame_index2 = torch.tensor(frame_indices_global2, dtype=torch.long)
        ### modify 0814, video_index_v vector embedding, 2, v
        ### smooth loss
        return clip, target, video_index, frame_index
        ### vector embedding
        # return clip, target, video_index_v, frame_index
        ### original
        # return clip, target, video_index, frame_index

    def __len__(self):
        return len(self.data)
