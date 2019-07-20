from .folder import ImageFolderInstance
from .cifar import CIFAR10Instance, CIFAR100Instance
from .mnist import MNISTInstance
from .ucf101 import UCF101Instance

__all__ = ('ImageFolderInstance', 'MNISTInstance', 'CIFAR10Instance', 'CIFAR100Instance')
