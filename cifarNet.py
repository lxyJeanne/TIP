import csv
import os
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.io import read_image
# Ignore warnings
import warnings
from torchsampler import ImbalancedDatasetSampler
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# class 1=>584, class 0=>32542
class MyNewDataset(Dataset):
    def __init__(self,label_csv,root_dir,transforms=None):
        self.img_labels=pd.read_csv(label_csv)
        self.root_dir=root_dir
        self.transforms=transforms

    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, item):
        img_path=os.path.join(self.root_dir,self.img_labels.iloc[item,0])
        img_path+='.jpg'
        imgs=read_image(img_path)
        label=self.img_labels.iloc[item,1]
        
        if self.transforms:
            imgs=self.transforms(imgs)
        
        return imgs,label



class MyDataset(Dataset):
    def __init__(self, x=None, y=None, filepath=None,transforms=None):
        if filepath:
            xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
            self.x = torch.from_numpy(xy[:, 1:])
            self.y = torch.from_numpy(xy[:, [0]])
            self.y = torch.flatten(self.y)
        else:
            self.x = x
            self.y = y
        self.len = self.x.shape[0] if self.x is not None else 0
        # total length
        print(f"length return= {self.len}")
        # find the right label order
        self.res = []
        self.list=self.y.numpy()
        self.list=self.list.tolist()
        self.list = list(map(int, self.list[:]))
        for j in self.list:
            if j not in self.res:
                self.res.append(j)
        print(f"class = {self.res}")
        print("Data prepared--------------------------------------------------------------------")

        # Set the transform for data augmentation
        self.transforms = transforms

    #prepare index: dataset[index]
    def __getitem__(self, index):
        x,y= self.x[index],self.y[index]

        # Reshape the flattened data to the original image shape
        x = x.view(3, 74, 74)  # Assuming images are 3-channel and size 74x74, adjust accordingly
        # Apply data augmentation only if the label is 1
        if y == 1 and self.transforms is not None:
            x = self.transforms(x)
            

        return x,y
    #prepare length: len(dataset)
    def __len__(self):
        return self.len

# Define data augmentation transform
augmentation_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    # Add other data augmentation operations
    transforms.ToTensor(),
])


# 创建一个函数来对少数类别进行过采样
def oversample_data(dataset):
    labels = dataset.label.numpy()

    # 使用 RandomOverSampler 对少数类别进行过采样
    oversampler = RandomOverSampler(sampling_strategy='minority', random_state=20)
    x_resampled, y_resampled = oversampler.fit_resample(dataset.image.numpy(), labels)

    # 转换为 PyTorch 张量
    x_resampled = torch.from_numpy(x_resampled)
    y_resampled = torch.from_numpy(y_resampled)
    print(f"x_reshaped={x_resampled.shape}")

    # 使用过采样后的数据创建新的数据集
    oversampled_dataset = MyNewDataset(image=x_resampled, label=y_resampled,filepath=None)
    return oversampled_dataset
    
"""   
labels_csv = 'testLabel.csv'
trainFolderPath='train-resizedtest'
trainImageData=process_images(trainFolderPath,labels_csv)
"""

label_csv = 'modified.csv'
trainFolder='real-resized'
#train_data_origin = MyDataset(filepath=train_file, transforms=augmentation_transform)
myData = MyNewDataset(label_csv=label_csv,root_dir=trainFolder,transforms=augmentation_transform)
#train_data = oversample_data(train_data_origin)

train_data,val_data=torch.utils.data.random_split(myData,[0.9,0.1])
#print(train_data)
#train_over_sampling_data=oversample_data(train_data)

# class 1=>584, class 0=>32542

#samples_weights = weights[train_targets]
#class_weights = [1/584, 1/32542]  # replace with your actual class weights
# Create sample weights from class weights


class_weights = [1/584, 1/32542]

# Get the labels for the training data
#y_train_indices = train_data.indices
#print(y_train_indices)
#print(len(train_data))
#print(train_data[1][])
#y_train = [train_data[i][1] for i in y_train_indices]

# Create sample weights from class weights
#samples_weight = [class_weights[t] for t in y_train]
#sampler_weights means the probability of each sample (for all 33126 samples)
# Create the sampler
#mySampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight.type('torch.DoubleTensor'), num_samples=len(samples_weight), replacement=True)
train_loader = DataLoader(dataset=train_data,
                          batch_size=25,
                          shuffle=True,
                          drop_last=True)


"""
test_file="test.csv"
test_data=MyNewDataset(label_csv=label_csv,root_dir=trainFolder)
"""
test_lst=[0,1]
val_loader=DataLoader(dataset=val_data,
                        batch_size=25,
                        shuffle=True,
                        drop_last=True)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Use_gpu = torch.cuda.is_available()
model = models.resnet50(pretrained=False)
print(f"Origine Model= {model}")

for param in model.parameters():
    param.requires_grad = True  # 原模型中的参数冻结，不进行梯度更新

model.fc=torch.nn.Linear(2048,2)
print(f"Modified Model= {model}")

if Use_gpu:
    model = model.to(device)


#Loss Fonction & Modulation of weight
weights = [1, 2]
class_weights = torch.FloatTensor(weights).to(device)
loss_func = nn.CrossEntropyLoss(weight=class_weights)
#loss_func = nn.CrossEntropyLoss()

#Learning rate=0.01
learning_rate=0.01
#Optimizer
optimizer_train=torch.optim.Adam(model.parameters(),lr=learning_rate)
# optimizer_train=torch.optim.Adam(cifar_train.parameters(),lr=learning_rate)
#Setting Steps
total_train_step=0
total_test_step=0

#Rounds for training
epoch=20

#data visualize: TensorBoard
writer=SummaryWriter("./logs_cifar")

for i in range(epoch):
    print("-----------------The {} round start-------------------".format(i+1))

    #test result total of every epoch
    final_predict=[]
    final_true=[]
    #start training
    for j,data in enumerate(train_loader):
        imgs,targets=data
        imgs,targets=imgs.to(device),targets.to(device)
        # imgs=imgs.reshape(20,3,74,74)  #batch,channel,img.height, img.weight
        outputs=model(imgs)
        # print(f"outputs: {outputs}")
        # print(f"targets: {targets}")

        #Add class weight to loss function
        loss=loss_func(outputs,targets.long())
        # print(f"loss={loss}")

        #optimizer
        optimizer_train.zero_grad()
        loss.backward()
        optimizer_train.step()
        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("Train step {}, Loss: {}".format(total_train_step,loss))
            # writer.add_scalar("train_loss",loss.item(),total_train_step)

    #start testing
    # Create model
    target_name=[str(i) for i in test_lst]
    # print(f"target_name= {target_name}")
    with torch.no_grad():    #ensure this data would not be optimized
        for l,data in enumerate(val_loader):
            imgs,targets=data
            imgs,targets=imgs.to(device),targets.to(device)

            outputs=model(imgs)
            # outputs = cifar_train(imgs)

            #loss=loss_func(outputs)
            outputs_np=outputs.cpu().numpy()
            #Decide the prediction of outputs --[predict probability of each class]
            x_predict=np.argmax(outputs_np,axis=1)
            x_predict=x_predict.tolist()
            #Get the true labels
            targets_np=targets.cpu().numpy()
            targets_np=targets_np
            targets_ls=targets_np.tolist()
            targets_ls=list(map(int,targets_ls[:]))
            y_true = list()
            for q in targets_ls:
                t=test_lst.index(q)
                y_true.append(t)

            #Sum up all batches of label lists
            final_true.extend(y_true)
            final_predict.extend(x_predict)

    #print(f'final predict label list= {final_predict}')
    #print(f'final true label list= {final_true}')
    #report table fonction
    report = classification_report(final_true, final_predict)
    print(report)
    # save trained model
    # torch.save(cifar_train,"cifar_{}.pth".format(i))
    # print("Model saved...")

writer.close()
