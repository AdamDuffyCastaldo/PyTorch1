
import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, features, img_size, emb_size=128):
        super().__init__()
        self.img_size = img_size
        bottleneck_size = self.img_size[-1]//(2**5)
        

        self.encoder = nn.Sequential(

            nn.Conv2d(self.img_size[0], features*8, (3,3), (1,1), 1), #in_channels = 3 for color, outchannels = 128, kernel_size(3,3), stride(1,1), padding =1 
            nn.BatchNorm2d(features*8), #batch normalisation
            nn.ReLU(),
            

            self._enc_block(features*8 , features*16),
            self._enc_block(features*16, features*16),
            self._enc_block(features*16, features*32),
            self._enc_block(features*32, features*16),
            self._enc_block(features*16, features*16),
             
              #256x4x4

            nn.Flatten(),
            nn.Linear(features*16*bottleneck_size*bottleneck_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, emb_size),
            #nn.Softmax(dim=1), #Normalise

        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ReLU(),
            nn.Linear(64,2, bias=False),
            nn.Softmax(dim=1)
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
            nn.ReLU(),
            nn.Dropout(0.1),
        )


    def forward(self, images):
        return self.encoder(images)
    
    def isSame(self, image1, image2, threshold=0.01):

        
        emb1 = self(image1)
        emb2 = self(image2)
        return self.emb_match(emb1, emb2)
    

    def emb_match(self, emb1, emb2):
        mse = F.mse_loss(emb1, emb2, reduction="none")
        similarity_score = self.classifier(mse)
        max_similarity_score, max_position = similarity_score.max(dim=1)  # Find the maximum value and its position
 
 
        return max_position