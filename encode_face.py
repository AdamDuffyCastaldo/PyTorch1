import torch
import torch.nn
import cv2
import sys
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

from model import Encoder
from imagecleaner import StraightenImage, resize, Load_image


class StraightenTransform(object):
    def __call__(self, x):
        image = np.array(x)
        image = StraightenImage(image)
        #print(image.shape)
        if image is None or image.shape[0] < 50 or image.shape[1] < 50:
            return x
        else:
            return image


def get_face_tensor(imgfile, device):
    image = Load_image(imgfile)

    transform = transforms.Compose([
    StraightenTransform(),
    transforms.ToTensor(),
    transforms.Pad(3), # adds 6 to both the height and width to make it 256,256
    transforms.Resize((128,128))
])

    tensorface = transform(image)
    return tensorface




# def RunModel(face2):
#     features = 18
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     tensorface = get_face_tensor(face2, device).to(device)
#     img_size = tensorface.shape
#     model = Encoder(features, img_size).to(device)
#     model.load_state_dict(torch.load("Data\Saved_model.pth"))
#     model.eval()
#     img = tensorface
#     if img == None:
#         print(-1)
#         return 1
#     with torch.no_grad():
#         embedding = model(img.unsqueeze(0))
#         return embedding

    # value = model.embmatch(embeddingface, embedding1)
    # return value

# class RunModel():
#     def __init__(self, features=18, model_path="Data\Saved_model.pth"):
#         self.features = features
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model_path = model_path
#         self.model = None
    
#     def Load_model(self, img):
#         self.img_size = get_face_tensor(img, self.device).shape
#         self.model = Encoder(self.features, self.img_size)
#         self.model.load_state_dict(torch.load(self.model_path))
#         self.model.eval()

#     def run_model(self, face_image):
#         if self.model is None:
#             print("Model not loaded. Please call load_model() first.")
#             return None
        
#         tensorface = get_face_tensor(face_image, self.device).to(self.device)
#         if tensorface is None:
#             print("Unable to process face image.")
#             return None
        
#         with torch.no_grad():
#             embedding = self.model(tensorface.unsqueeze(0))
#         return embedding



def compareImages(face1, face):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb10 = get_face_tensor(face1, device).to(device)
    emb20 = get_face_tensor(face, device).to(device)
    
    

    features = 18
    # embedding2 = RunModel(face)
    
    img_size = emb10.shape
    model = Encoder(features, img_size).to(device)
    model.load_state_dict(torch.load("Data\Saved_model.pth"))
    model.eval()
    emb1 = model(emb10.unsqueeze(0))
    emb2 = model(emb20.unsqueeze(0))

    is_same_prediction = model.isSame(emb10.unsqueeze(0), emb20.unsqueeze(0))

    # emb1 = F.normalize(emb1, p=2, dim=1)
    # emb2 = F.normalize(emb2, p=2, dim=1)
    # similarity = torch.sum(emb1*emb2, dim=1)
    # #print(similarity)
    # is_same = similarity > 0.7
    return is_same_prediction


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image1 = "adamba46ca7-4ecd-4aca-98bf-936ba114ffcd.jpg"
image2 = "images\download (3).jpg"

issamepredict = compareImages(image1, image2)

if issamepredict.item() == 1:
    issamepredict = True
else:
    issamepredict = False


print(issamepredict)


