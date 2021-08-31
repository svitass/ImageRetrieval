from torch.utils.data import Dataset
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self,path,transform=None): # path: .../train    .../val
        super(MyDataset, self).__init__()
        self.query_imgs = []
        self.shop_p_imgs = []
        self.shop_n_imgs = []
        assert os.path.exists(path) , "{} path does not exist.".format(path)
        for triplet_path in os.listdir(path):
            img_path = os.path.join(path,triplet_path)
            # 保存图片路径  直接读取图片，会导致内存占用过高，然后导致程序崩溃
            q_img = os.path.join(img_path,'query.jpg')
            p_img = os.path.join(img_path,'relevant.jpg')
            n_img = os.path.join(img_path,'nonRelevant.jpg')
            self.query_imgs.append(q_img)
            self.shop_p_imgs.append(p_img)
            self.shop_n_imgs.append(n_img)
        self.transform = transform

    def __getitem__(self, index):
        q_path = self.query_imgs[index]
        p_path = self.shop_p_imgs[index]
        n_path = self.shop_n_imgs[index]
        q = Image.open(q_path).convert('RGB')
        p = Image.open(p_path).convert('RGB')
        n = Image.open(n_path).convert('RGB')
        if self.transform:
            q = self.transform(q)
            p = self.transform(p)
            n = self.transform(n)
        return p,q,n

    def __len__(self):  # p,q,n成对出现，长度一样
        return len(self.query_imgs)

