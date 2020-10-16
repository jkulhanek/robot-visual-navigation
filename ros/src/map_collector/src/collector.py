from common import Proxy
import cv2
import os
import io
from PIL import Image

PATH = "/mnt/data/"
class CollectCache(object):
    def __init__(self, *args, **kwargs):
        self.initialize()
        super(CollectCache, self).__init__(*args, **kwargs)

    def initialize(self):
        if not os.path.exists(PATH + "dataset/images"):
            os.makedirs(PATH + "dataset/images")
            os.makedirs(PATH + "dataset/depths")
        self.lastIndex = len([name for name in os.listdir(PATH + 'dataset/images') if os.path.isfile(os.path.join(PATH + 'dataset/images', name))])


    def collect_observation(self, observation, position, rotation):
        index = self.lastIndex + 1

        Image.fromarray(observation[0]).save(PATH + "dataset/images/%s.png" % index)
        Image.fromarray(observation[1]).save(PATH + "dataset/depths/%s.png" % index)
        with open(PATH + "dataset/info.txt", "a+") as f:
            f.write(u"%s %s %s %s %s %s %s %s %s %s\n" % ((index,) + (position[0], position[1], rotation,) + (observation[2][0], observation[2][1], observation[3][2]) + observation[4]))
            f.flush()
        self.lastIndex = index
        pass

cache = CollectCache()

def make_file_dataset(navigator):
    oldCollect = navigator.collect
    def collect(observation, position, rotation, *args, **kwargs):
        oldCollect(observation, position, rotation, *args, **kwargs)
        cache.collect_observation(observation, position, rotation)
    navigator.collect = collect
    return navigator

    # class NavigatorProxy(Proxy):
    #     def __init__(self, *args, **kwargs):
    #         super(NavigatorProxy, self).__init__(*args, **kwargs)

    #     def __getattribute__(self, name):
    #         if name == "collect":
    #             return collect
    #         return super(NavigatorProxy, self).__getattribute__(name)
        
    # return NavigatorProxy(navigator)
    