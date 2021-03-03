import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from model import resnet

label = 0

def test(imagepath):
    print(imagepath)
    res1 = 0
    res2 = 0
    res3 = 0
    imgcnt = 0
    for root, subdirs, imgs in os.walk(imagepath):
        print(imagepath)
        for img in imgs:
            print(img)
            img_path = os.path.join(imagepath, img)
            arch = 'resnet50'
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            valid_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
            '''
            preprocess_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
            '''
            model = torch.nn.DataParallel(resnet.__dict__[arch]())
            PATH = "./save_temp/resnet/model.th"
            model.load_state_dict(torch.load(PATH)["state_dict"])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            image = Image.open(img_path).convert('RGB')
            image_tensor = valid_transform(image)
            image_tensor.unsqueeze_(0)
            image_tensor = image_tensor.to(device)
            out = model(image_tensor)
            _, indices = torch.sort(out, descending=True)
            # print(indices[0][0].item(), end='')
            if indices[0][0].item() == 0:
                res1 += 1
            elif indices[0][0].item() == 1:
                res2 += 1
            elif indices[0][0].item() == 2:
                res3 += 1
            imgcnt += 1

    imagepath = './testdata'
    print('\n')
    print("P:" + str(res1) + '/' + str(imgcnt))
    print("N:" + str(res2) + '/' + str(imgcnt))
    print("C:" + str(res3) + '/' + str(imgcnt))
    if (res1/imgcnt)>0.2 and res1>res3:
        return 0
    elif (res3/imgcnt)>0.2 and res3>res1:
        return 2
    else:
        return 1


class CovidDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"P": 0, "N": 1, "C": 2}
        self.data_info = self.getImgInfo(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')  # 0~255

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def getImgInfo(data_dir):
        data_info = list()
        data_info.append((data_dir, int("2")))

        return data_info


if __name__ == '__main__':
    main()
