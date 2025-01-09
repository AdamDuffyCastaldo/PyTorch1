This repo was made just to test out PyTorch and how to normalise images for input to a model. Code to detect if two images are the same are also here.

Explanation of each (important) file: model.py

the architecture of the CNN pytorch.ipynb
Where I trained the model with a dataset, including all testing of the models parameters, weights etc. Accuracy/metrics of both validation set and dataset are in here. facedetector.py
This allows me to input an image, and have it straigtened. This was made so that tilted and askew faces could still be used. imagecleaner.py
takes an input image, formats it, straigtens it and resizes it for input to model. encode_face.py
This takes two images, and uses the imagecleaner to first normalise the two images and then uses the model to find out if they are the same or not, returning true or not. saved_model.pth
The weights and parameters exported for use in my project.
