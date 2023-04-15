import os
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

cmap = [[0,    0,    0], [128,    0,    0], [0,  128,    0], [128,  128,    0], [0,    0,  128], [128,    0,  128], [0,  128,  128], [128,  128,  128], [64,    0,    0], [192,    0,    0], [
            64,  128,    0], [192,  128,    0], [64,    0,  128], [192,    0,  128], [64,  128,  128], [192,  128,  128], [0,   64,    0], [128,   64,    0], [0,  192,    0], [128,  192,    0], [0,   64,  128]]
# cmap=list(map(lambda x:list(map(lambda y:y/256,x)),_cmap))
transform = transforms.Compose([ # 一般用Compose把多个步骤整合到一起
    transforms.ToTensor()
])
PILtransform=transforms.Compose([transforms.PILToTensor()])
# 将dataset类中__getitem__()方法内读入的PIL或CV的图像数据转换为torch.FloatTensor


class onehot(object): #编码器
    def __init__(self):
        self.n_classes = 21
        
    def __call__(self, image):
        h0,w0=image.size
        image_tensor=PILtransform(image).float()
        c,h,w=image_tensor.shape
        # save_image(image_tensor/255,'C:/users/xiezh/desktop/0.png')
        # se0=set({})
        # for i in range(h0):
        #     for j in range(w0):
        #         se0.add(image.getpixel((i,j)))
        # print(se0)
        image_tensor=image_tensor.permute(1,2,0)
        # se=set({})
        # for i in range(h):
        #     for j in range(w):
        #         se.add(tuple(image_tensor[i,j].long().numpy().tolist()))
        # print(se)

        mask=[]
        for color in cmap:
            color=torch.tensor(color)
            equality=torch.all(torch.eq(image_tensor.long(),color),dim=-1)
            mask.append(equality)
        mask=torch.stack(mask,axis=0)
        return mask.long().float().cuda()

class onehotdecoder(object): #解码器
    def __init__(self):
        self.n_classes=21
    
    def batchtoclass(self, tensor_image):
        return torch.argmax(tensor_image,dim=1)
    
    
    def __call__(self, tensor_image):
        c,h,w=tensor_image.size()
        class_image=torch.argmax(tensor_image,dim=0,keepdim=False)
        # class_image.squeeze_()
        r=torch.zeros(class_image.shape)
        g=torch.zeros(class_image.shape)
        b=torch.zeros(class_image.shape)
        # print(class_image)
        for i in range(21):
            pos=class_image==i
            r[pos]=cmap[i][0]
            g[pos]=cmap[i][1]
            b[pos]=cmap[i][2]
        rgb=torch.stack([r,g,b],dim=0).float()
        rgb=rgb/255 #归一化
        # print(f"rgb:{rgb.any()}")
            # class_image[class_image==torch.tensor([i])]=torch.tensor(cmap[i])
        rgb=rgb.cuda()
        return rgb


class MyDataset(Dataset):
    def __init__(self, path, mode):
        self.path = path
        f=open(os.path.join(path,f'ImageSets\\Segmentation\\{mode}.txt'),"r")
        self.name=list(map(str.strip,f.readlines()))
        f.close()
        self.onehot_encoder=onehot()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # Note:原始数据中的JPEGImages比SegmentationClass图片多很多张，这个代码中只用了SegmentationClass图片去检索JPEGImages原图
        segment_name = self.name[index]+'.png'  ## xx.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))  # 原图地址
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)  # 原图等比缩放
        seg=self.onehot_encoder(segment_image)
        return transform(image),seg


if __name__ == '__main__':
    data = MyDataset(r'E:\Pascal VOC 2012\VOCdevkit\VOC2012','train') # Python在windows下的标准路径是斜杠’ / ’ ,但是仍然可以识别 反斜杠’ \ ’，如果路径中有'\Uxxx'或者'\txxx'的时候会识别不出来，建议使用规范的路径写法'/'
    oh=onehot()
    de=onehotdecoder()
    data1=torch.tensor([[[0,128,128],[0,128,128],[128,128,0],[192,0,128],[255,255,255]],[[0,0,128],[128,0,128],[0,0,0],[192,128,0],[64,128,128]],[[0,0,128],[128,0,128],[0,0,0],[255,255,255],[64,128,128]]])
    data1=data1.permute(2,0,1)
    tf=transforms.ToPILImage()
    # print(data1)
    # print(data[0][1][1:].any())
    img=de(data[0][1]) 
    # print(data1.numpy())
    pil1=Image.fromarray(data1.numpy(),mode='RGB')
    w,h=pil1.size
    # for i in range(w):
    #     for j in range(h):
    #         print(pil1.getpixel((i,j)),end=" ")
    #     print("")
    # pil1.convert("P")
    # for x,y in pil1:
    #     (x,y))
    # print("oh",oh(data1))
    # print("de",(de(oh(tf((data1/255).float())))*255).permute(1,2,0))
    # print((img>0).any())
    save_image(img, 'C:/users/xiezh/desktop/1.png')
    # print(oh(data[0][1]).shape)  # 打印第1张图的分割图形状






