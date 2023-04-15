from PIL import Image
import torchvision.transforms as transforms
# Python图像库PIL(Python Image Library)是python的第三方图像处理库，但是由于其强大的功能与众多的使用人数，几乎已经被认为是python官方图像处理库了。


def keep_image_size_open(path, size=(256,256)):
    '''
    # 等比缩放
    :return:
    '''
    img = Image.open(path)
    md=img.mode
    temp = max(img.size) # 图片最长边
    mask = Image.new(md, (temp, temp)) # 用图片最长边做一个矩形，黑色
    w,h=img.size
    se=set({})
    # for i in range(h):
    #     for j in range(w):
    #         se.add(img.getpixel((j,i)))
    # print(se)

    mask.paste(img, (0,0)) # 粘贴到左上角
    mask = mask.resize(size)
    if md=='P':
        mask.putpalette(img.palette)
    mask=mask.convert("RGB")
    return mask

