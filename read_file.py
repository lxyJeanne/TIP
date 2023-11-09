import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Spécifier le chemin du dossier de l'image
image_folder = 'train-resized'
label_list = 'train-labels.csv'
label_df = pd.read_csv(label_list)

# 创建一个空的DataFrame来存储所有图片数据
image_data_df = pd.DataFrame(columns=['filename','label', 'image_data'])

# 遍历图像文件夹并处理每张图片
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        img = plt.imread(image_path)
        # 将图像转换为PyTorch张量
        img_tensor = transforms.ToTensor()(np.array(img))
        name = filename.rsplit('.', 1)[0]
        label_row = label_df[label_df['image_name'] == name]

        df = pd.DataFrame({'filename': [name],'label': label_row['target'], 'image_data': [img_tensor]})
        # 将图片数据和文件名添加到DataFrame
        image_data_df =pd.concat([image_data_df,df])

# 保存DataFrame到CSV文件
image_data_df.to_csv('train_data.csv', index=False)

