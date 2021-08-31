import os
from shutil import copy,rmtree
import shutil


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    cwd = os.getcwd()
    root_path = os.path.join(cwd,'img_highres-002','img_highres')
    train_path = os.path.join(cwd,'img_highres-002','train')
    val_path = os.path.join(cwd,'img_highres-002','val')
    i = 1
    for clothkinds in os.listdir(root_path):
        cloth_path = os.path.join(root_path,clothkinds)
        print('cloth_path:',cloth_path)
        for kinds in os.listdir(cloth_path):
            kinds_path = os.path.join(cloth_path,kinds)
            print('kinds_path:' , kinds_path)
            pairs = os.listdir(kinds_path)
            length = len(pairs)
            # query image:pair1里的consumer_01.jpg  relevant image:pair1里的shop_01.jpg  non-relevant image:pair2里的shop_02.jpg
            # 选择non-relative image为同一小类里且与query image不在一个文件夹里的图片
            for j in range(0,length-10,2):  # 每个种类的后10个用作测试数据
                pair1_path = pairs[j]    # 获取图像对所在的文件夹
                pair2_path = pairs[j+1]
                query_img_path = os.path.join(kinds_path,pair1_path,'comsumer_01.jpg')
                relevant_img_path = os.path.join(kinds_path,pair1_path,'shop_01.jpg')
                nonRelevant_img_path = os.path.join(kinds_path,pair2_path,'shop_01.jpg')
                if (not os.path.exists(query_img_path)) or (not os.path.exists(relevant_img_path)) or (not os.path.exists(nonRelevant_img_path)):
                    continue
                # 为新的三元组(q,p,n)创建一个文件夹
                if i % 7 != 0:  # train
                    path = os.path.join(train_path,str(i).zfill(6))
                else:  # val
                    path = os.path.join(val_path,str(i).zfill(6))
                mk_file(path)
                new_q_path = os.path.join(path,'query.jpg')
                new_p_path = os.path.join(path,'relevant.jpg')
                new_n_path = os.path.join(path,'nonRelevant.jpg')
                copy(query_img_path,new_q_path)
                copy(relevant_img_path,new_p_path)
                copy(nonRelevant_img_path,new_n_path)
                i = i + 1
        print(kinds)
    print("processing done!")


if __name__ == '__main__':
    main()
