
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import transforms


class MyDataset(Dataset):
    def __init__(self,filepath):

        xy=np.loadtxt(filepath, delimiter=',',dtype=np.float32)

        #total length
        self.len = xy.shape[0]

        self.x=xy[:,1:]

        self.x=torch.from_numpy(self.x)
        print(f"read data= {self.x}")
        print(f"read data shape= {self.x.shape}")

        #create y_label
        self.y = xy[:, [0]]
        self.y=torch.from_numpy(self.y)
        self.y=torch.flatten(self.y)
        print(f"read label = {self.y}")
        print(f"read label shape= {self.y.shape}")

        # find the right label order
        self.res = []
        self.list=self.y.numpy()
        self.list=self.list.tolist()
        self.list = list(map(int, self.list[:]))
        for j in self.list:
            if j not in self.res:
                self.res.append(j)
        print(f"class = {self.res}")
        print("Data prepared------------")

    #prepare index: dataset[index]
    def __getitem__(self, index):
        return self.x[index],self.y[index],self.res

    #prepare length: len(dataset)
    def __len__(self):
        return self.len

train_file="train.csv"
train_data=MyDataset(train_file)
train_lst=train_data.res
train_loader=DataLoader(dataset=train_data,
                        batch_size=10,
                        shuffle=True,
                        drop_last=True)
test_file="test.csv"
test_data=MyDataset(test_file)
test_lst=test_data.res
# print(test_lst)
test_loader=DataLoader(dataset=test_data,
                        batch_size=10,
                        shuffle=True,
                        drop_last=True)

# construction of model
class CIFAR(nn.Module):
    def __init__(self,class_num):
        super(CIFAR, self).__init__()
        self.class_num=class_num
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(5184,64),
            nn.Linear(64,self.class_num)
        )

    def forward(self, x):
        x = self.model(x)
        return x


#Loss Fonction
loss_func=nn.CrossEntropyLoss()

#Optimizer
#Learning rate=0.01
learning_rate=1e-2

#Setting step parameters
total_train_step=0
total_test_step=0
#turns for training
epoch=2


#data visualize: TensorBoard
# writer=SummaryWriter("../logs_cifar")
#Create model
cifar_train=CIFAR(len(train_lst))
optimizer_train=torch.optim.SGD(cifar_train.parameters(),lr=learning_rate)
for i in range(epoch):
    print("-----------------The {} round start-------------------".format(i+1))

    #test result total of every epoch
    final_predict=[]
    final_true=[]
    #start training
    for j,data in enumerate(train_loader):
        imgs,targets,l=data
        imgs=imgs.reshape(10,3,74,74) #batch,channel,img.height, img.weight
        outputs=cifar_train(imgs)
        # print(f"outputs: {outputs}")
        # print(f"targets: {targets}")
        loss=loss_func(outputs,targets.long())
        print(f"loss={loss}")
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
    print(f"target_name= {target_name}")
    with torch.no_grad():
    #ensure this data would not be optimized
        for l,data in enumerate(test_loader):
            imgs,targets,k=data
            imgs = imgs.reshape(10,3,74,74)
            outputs=cifar_train(imgs)
            loss=loss_func(outputs,targets.long())
            # _,predict=torch.max(outputs,1)
            outputs_np=outputs.numpy()
            # print(outputs_np)
            x_predict=np.argmax(outputs_np,axis=1)
            x_predict=x_predict.tolist()
            print(f"label_predict= {x_predict}")

            targets_np=targets.numpy()
            targets_np=targets_np
            targets_ls=targets_np.tolist()
            targets_ls=list(map(int,targets_ls[:]))
            y_true = list()
            for q in targets_ls:
                t=test_lst.index(q)
                y_true.append(t)
            print(f"label true= {y_true}")

            final_true.extend(y_true)
            final_predict.extend(x_predict)

    print(f'final predict label list= {final_predict}')
    print(f'final true label list= {final_true}')
    report = classification_report(final_true, final_predict)
    print(report)
        #save trained model
        #torch.save(cifar,"cifar_{}.pth".format(i))
        #print("Model saved...")

# writer.close()
