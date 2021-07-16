import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from corruptions import *
from typing import Callable, Optional, Tuple, Any
from torchvision import transforms

class CorruptDataset(ImageFolder):
    def __init__(self, root: str, corruption: str, intensity:int, transform: Optional[Callable] = None):
        super(CorruptDataset, self).__init__(root = root, transform = transform)
        self.corruption = corruption
        self.intensity = intensity
        self.transform = transform
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, _ = self.samples[index]
        pil_img = Image.open(img)
        pil_img = pil_img.convert('RGB')

        if self.corruption and self.intensity is not None:
            if type(self.corruption) == list and type(self.intensity) == list:
                ret = [torchvision.transforms.ToTensor()(pil_img)]
        
                corr_intensity_lst = zip(self.corruption, self.intensity)

                for corruption, intensity in corr_intensity_lst:
                    if corruption in weather:
                        np_img = np.array(pil_img)
                        ret.append(transforms.ToTensor()(key2deg[corruption](np_img, int(intensity))).type(torch.float))
                    else:
                        img = torchvision.transforms.ToTensor()(pil_img)
                        ret.append(torch.tensor(key2deg[corruption](img, int(intensity))).type(torch.float))

                return ret 
            else:
                raise ValueError('Not a list')
        else:
            return tnsr_img