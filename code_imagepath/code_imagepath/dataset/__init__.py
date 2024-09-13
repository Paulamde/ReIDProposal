# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .VR import VR
from .VR_sy import VR_sy
from .VR_test import VR_test
from .VeRi_test import VeRi_test
from .VeRi import VeRi
from .VeRi_sy import VeRi_sy
from .VeRi_sy_test import VeRi_sy_test
from .VeRi_sy_test import VeRi_sy_test
from .VR_sy_test import VR_sy_test
from .SyNew_test import SyNew_test
from .SyNew import SyNew

from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'VR': VR,
    'VR_sy': VR_sy,
    'VR_sy_test': VR_sy_test,
    'VR_test': VR_test,
    'VeRi_test': VeRi_test,
    'VeRi': VeRi,
    'VeRi_sy_test': VeRi_sy_test,
    'VeRi_sy': VeRi_sy,
    'SyNew': SyNew,
    'SyNew_test': SyNew_test,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
