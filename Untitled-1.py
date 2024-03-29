# %%
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image


import matplotlib.pyplot as plt
from tqdm import tqdm



# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%

VAL_RATIO = 0.2

#The ratio of the total training set used only for validation

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(3) # adds 6 to both the height and width to make it 256,256
])



dataset = torchvision.datasets.LFWPairs(root = "PyTorch\\",download="True",split="train", transform=transform)

INPUT_SHAPE = (3,250,250)
VAL_RATIO = 0.2
val_len = int(len(dataset) * VAL_RATIO)
train_len = len(dataset) - val_len

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

print(val_len,train_len)

image1, image2, label =  train_dataset[0]
image1, image2 = image1.to(torch.device("cuda")), image2.to(torch.device("cuda"))
image1.shape, image2.shape, label

# %%

len(train_dataset), len(train_dataset), len(val_dataset), len(val_dataset)

# %%
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
idxs = torch.randperm(len(train_dataset))[:10] #10 random images
for i, ax in enumerate(axes):
    img, img2, label = train_dataset[idxs[i]]
    ax.imshow(img.cpu().permute(1,2,0)) #moves dimensions 3,250,250 --> 250,250,3
    ax.axis('off')
plt.show()

# %%
BATCH_SIZE = 127

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device 

# %%
class Encoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.img_size = (3,256,256)

        self.encoder = nn.Sequential(

            nn.Conv2d(self.img_size[0], features*8, (3,3), (1,1), 1),
            nn.BatchNorm2d(features*8),
            nn.ReLU(),

            self._enc_block(features*8 , features*16),
            self._enc_block(features*16, features*16),
            self._enc_block(features*16, features*32),
            self._enc_block(features*32, features*16),
            self._enc_block(features*16, features*16),
             
              #256x4x4

            nn.Flatten(),
            nn.Linear(features*16*8*8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            #nn.Softmax(dim=1), #Normalise

        )


    def _enc_block(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(out_channels), #normalises data
            nn.ReLU6(),
            nn.Dropout(0.1),
        )

    def forward(self, images):
        return self.encoder(images)
    

# %%
batchsize = 16

image1.shape


# %%
batchsize = 16




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Encoder(8)
model = model.to(device)


optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
lossarray = []

# %%
batchsize = 16
num_epochs = 10
traindataloader = DataLoader(train_dataset, batchsize, shuffle=True)
for epoch in range(num_epochs):
    loop = tqdm(traindataloader, leave=False, total = len(traindataloader))
   
    if epoch > 0:
        loop.set_description(f"Epoch : {epoch}/{num_epochs}")
        loop.set_postfix({"Loss" : lossarray[-1]})
    
    batchlosses = []

    for (images1, images2, labels) in loop:
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)
        out1 = model(images1)
        out2 = model(images2)
        #print(out1, out2)
        labels = (labels - 0.5)*2 
        loss = F.mse_loss(out1, out2)*label # mean squared error calculates the distance between the 2 images and then squares it to take the absolute distance between them. This will produce a vector with the distance between each weight in the node
        print(loss)
        loss.backward() # backward needs to be done on a vector object and mean reduces the dimensions of the loss.
        batchlosses.append(loss.item()) #gives the value rather than the object

    lossarray.append(sum(batchlosses)/len(batchlosses))





# %%
#write comparison script between faces
#function that takes 2 faces as input and returns same or not.
# vector output -- 128 in length, vector difference and determine threshold
#saves state dict



