import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import os
from Siamese3 import Siamese

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:' , device)

    data_transform = transforms.Compose(
        [transforms.Resize(256) ,
         transforms.CenterCrop(224) ,
         transforms.ToTensor() ,
         transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])])  # 和训练一样

    # load image
    data_root = os.path.abspath(os.path.join(os.getcwd()))
    image_path = os.path.join(data_root , 'data_set' , 'img_highres-002','test')
    assert os.path.exists(image_path) , "{} path does not exist.".format(image_path)
    # 获取query image,non-relevant image,shop image(shop image假设作为relevant image)
    q = Image.open(os.path.join(image_path,'query','comsumer_01.jpg'))
    n = Image.open(os.path.join(image_path,'nonRelevant','nonrelative.jpg'))
    q = torch.unsqueeze(data_transform(q),dim=0).to(device)
    n = torch.unsqueeze(data_transform(n),dim=0).to(device)

    shop_img_path = os.path.join(image_path,'shop')


    # create model
    net = Siamese().to(device)

    # load model weights
    weights_path = "./Siamese.pth"
    assert os.path.exists(weights_path) , "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path,map_location=device))

    # prediction
    predict_path = ''
    minLoss = float('inf')
    net.eval()
    margin = 0.1
    shop_imgs = os.listdir(shop_img_path)
    shop_img_num = len(shop_imgs)
    i = 1
    loss_function = nn.TripletMarginLoss(margin=margin , p=2)
    with torch.no_grad():
        for p_path in os.listdir(shop_img_path):
            p = Image.open(os.path.join(shop_img_path,p_path))
            # show shop img
            plt.subplot(2,shop_img_num,i)
            plt.title('shop '+ str(i))
            plt.imshow(p)
            i += 1
            p = torch.unsqueeze(data_transform(p),dim=0).to(device)
            q_out , p_out , n_out = net(q , p , n)
            loss = loss_function(q_out , p_out , n_out)
            if minLoss > loss:
                minLoss = loss
                predict_path = p_path
    print('minLoss:',minLoss)
    result_img = Image.open(os.path.join(shop_img_path,predict_path))
    q = Image.open(os.path.join(image_path , 'query' , 'comsumer_01.jpg'))
    plt.subplot(2,shop_img_num,i+1)
    plt.title('query image')
    plt.imshow(q)

    plt.subplot(2,shop_img_num,i+2)
    plt.title('retrieval result')
    plt.imshow(result_img)

    plt.show()


if __name__ == '__main__':
    main()