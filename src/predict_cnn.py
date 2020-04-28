import torch
import torchvision.transforms as transforms
from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from models.UNet import UNet
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils import mean_dice, convert_mask_to_rgb_image
import warnings
import ntpath


def predict_image(model,img,use_cuda=True):
    device_name = 'cuda:0' if use_cuda else 'cpu'
    if use_cuda and not torch.cuda.is_available():
        warnings.warn("CUDA is not available. Suppress this warning by passing "
                        "use_cuda=False.")
        device_name = 'cpu'

    device = torch.device(device_name)
    # Since the model expect a 4D we need to add the batch dim in order to get the 4D
    img = np.expand_dims(img, axis=0)
    # convert image to Tensor
    img = torch.from_numpy(img)
    img = img.to(device)
    prediction = model(img)
    # delete the batch dimension
    prediction = np.squeeze(prediction)
    # convert prediction to numpy array
    # take into account if model is trained on cpu or gpu
    if use_cuda and torch.cuda.is_available():
        prediction = prediction.detach().cpu().numpy()
    else:
        prediction = prediction.detach().numpy()
    # from one_hot vector to categorical
    prediction = np.argmax(prediction, axis=0)
    # convert the predicted mask to rgb image
    prediction = convert_mask_to_rgb_image(prediction)
    # remove the batch dim and the channel dim of img
    img = np.squeeze(np.squeeze(img))
    # convert img to a numpy array
    if use_cuda and torch.cuda.is_available():
        img = img.cpu().numpy()
    else:
        img = img.numpy()

    return prediction

model = UNet(num_classes=4, in_channels=3)
model.load_weights('UNet.pt')

if len(sys.argv) > 1 :
    use_cuda = ( sys.argv[1] == 'True' )
else :
    use_cuda = True

print("model loaded!")

device_name = 'cuda:0' if use_cuda else 'cpu'
if use_cuda and not torch.cuda.is_available():
    warnings.warn("CUDA is not available. Suppress this warning by passing "
                    "use_cuda=False.")
    device_name = 'cpu'

device = torch.device(device_name)

model = model.to(device)

# if len(sys.argv) == 1 :
#     print("veuillez donnez le chemin de l'image à prédire en ligne de commande")
#     exit()

# img_path = sys.argv[1]

img_path = None

while(True):
    if img_path == None :
        img_path = input("entre img path (enter to exit) : ")
        if len(img_path) == 0:
            exit()
    img_name = ntpath.basename(img_path)

    img = Image.open(img_path).resize((960,720)).convert("RGB")
    original_size = img.size

    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    img = transform(img)
    prediction = predict_image(model,img)

    # plot the image
    f = plt.figure(figsize=(10, 10))
    ax1 = f.add_subplot(121)
    ax1.imshow(np.transpose(img, (1, 2, 0)))
    ax1.set_title('image')
    ax1.axis('off')

    # plot the ground truth
    ax2 = f.add_subplot(122)
    ax2.imshow(prediction.astype('uint8'))
    ax2.set_title('predicted segmentation')
    ax2.axis('off')


    # Save as a png image
    f.savefig('../prediction/predict_cnn_'+img_name)
    # show image
    plt.show()
    Image.fromarray(prediction.astype('uint8')).resize(original_size).save('../prediction/prediction_cnn_{}'.format(img_name))

    img_path = None

