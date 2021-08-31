import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms,datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from Siamese3 import Siamese
from MyDataset import MyDataset
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 多进程需要在main函数中运行，否则应改成单进程加载（num_workers=0）
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:',device)
    net = Siamese()
    net = net.to(device)
    # triplet loss
    margin = 0.1
    loss_function = nn.TripletMarginLoss(margin=margin,p=2)
    learning_rate = 0.0005
    # momentum:动量，加快学习过程，适用于小但一致的梯度或者噪声比较大的数据
    optimizer = optim.SGD(net.parameters() , lr=learning_rate , momentum=0.9)

    # 数据处理
    data_transform = {
        # 当Key为train时，返回训练集所使用的数据预处理的方法
        "train": transforms.Compose([transforms.RandomResizedCrop(224) ,  # 随机裁剪 224*224
                                     transforms.RandomHorizontalFlip() ,  # 随机翻转
                                     transforms.ToTensor() ,  # 转化为一个tensor
                                     transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])]) ,
    # 标准化处理
        "val": transforms.Compose([transforms.Resize(256) ,
                                   transforms.CenterCrop(224) ,
                                   transforms.ToTensor() ,
                                   transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd()))
    image_path = os.path.join(data_root , 'data_set' , 'img_highres-002')
    # ImageFolder是针对分类任务加载数据集
    train_dataset = MyDataset(path=os.path.join(image_path , "train") , transform=data_transform["train"])
    validate_dataset = MyDataset(path=os.path.join(image_path , "val") , transform=data_transform["val"])

    batch_size = 9  # 若为16,则CUDA内存超出
    # 线程个数,Windows下不能设置成一个非0值
    nw = 2
    # nw = min([os.cpu_count() , batch_size if batch_size > 1 else 0 , 8])
    train_loader = DataLoader(train_dataset , batch_size=batch_size , shuffle=False ,
                                               num_workers=nw)
    validate_loader = DataLoader(validate_dataset , batch_size=batch_size , shuffle=False ,
                                                  num_workers=nw)
    # 开始训练
    epochs = 3
    best_acc = 0.0
    save_path = './Siamese.pth'
    train_steps = len(train_loader)
    val_num = len(validate_dataset)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader)  # 进度条
        for i,data in enumerate(train_bar):
            # 获取q,p,n
            q,p,n = data
            # images,labels = data
            # # N是图片的张数，可同时多个图片，多个标签
            # q,p,n = images
            # q = torch.unsqueeze(q , dim=0).type(torch.FloatTensor) # 在第0维上增加维数为1的维度
            # p = torch.unsqueeze(p , dim=0).type(torch.FloatTensor)
            # n = torch.unsqueeze(n , dim=0).type(torch.FloatTensor)
            q = q.to(device)
            p = p.to(device)
            n = n.to(device)
            # 前向传播
            q_out,p_out,n_out = net(q,p,n)   # triplet_loss(anchor, positive, negative)
            loss = loss_function(q_out,p_out,n_out)
            running_loss += loss.item()
            # # output = torch.max(input,dim) input是一个tensor,dim 0表示每列的最大值，1表示每行的最大值
            # _,pred = torch.max(out,1)  # 预测结果 返回两个tensor：最大值，最大值的索引
            # num_correct = (pred==labels).sum() #正确结果的数量
            # running_acc += num_correct.item()  # 正确结果的总数

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 后向传播计算梯度
            optimizer.step()  # 利用梯度更新W.b参数

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1 ,
                                                                     epochs ,
                                                                     loss)
        # validate
        net.eval()
        loss = 0
        acc = 0.0
        with torch.no_grad():  # 强制之后的内容不进行计算图构建
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                q , p , n = val_data
                q = q.to(device)
                p = p.to(device)
                n = n.to(device)
                q_out , p_out , n_out = net(q , p , n)
                loss = loss_function(q_out , p_out , n_out)
                #loss >= margin: #预测错误--认为q和n更接近
                if loss < margin: # 预测正确---认为q和p更接近
                    acc += 1
                # _ , predict = torch.max(outputs , dim=1)
                # acc += torch.eq(predict , labels).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1 ,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1 , running_loss / train_steps , val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict() , save_path)  # 保存模型参数

        print('Finished Training')


if __name__ == '__main__':
    main()
