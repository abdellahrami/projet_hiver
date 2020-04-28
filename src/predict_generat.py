import cv2
from plantcv import plantcv as pcv
from PIL import Image
import sys
from matplotlib import pyplot as plt
import numpy as np
import ntpath

# if len(sys.argv) > 1 :
#     print("veuillez donnez le chemin de l'image à prédire en ligne de commande")
#     exit()

# img_path = sys.argv[1]
img_path = None

class options:
    def __init__(self,path_img):
        self.image = path_img
        self.debug = None
        self.writeimg= False 
        self.outdir = "."

while(True):
    if img_path == None :
        img_path = input("entre img path (enter to exit) : ")
        if len(img_path) == 0:
            exit()
    img_name = ntpath.basename(img_path)
    # Get options
    args = options(img_path)
    img_path = None

    # Set debug to the global parameter 
    pcv.params.debug = args.debug

    img, path, filename = pcv.readimage(filename=args.image)

    # Inputs: 
    #   rgb_img - RGB image data 
    #   pdf_file - Output file containing PDFs from `plantcv-train.py`
    mask = pcv.naive_bayes_classifier(rgb_img=img, 
                                    pdf_file="../data/processed/naive_bayes_pdfs.txt")


    # plot the image
    f = plt.figure(figsize=(10, 10))
    ax1 = f.add_subplot(121)
    ax1.imshow(img)
    ax1.set_title('image')
    ax1.axis('off')

    # plot the ground truth
    ax2 = f.add_subplot(122)
    ax2.imshow(mask['Arbre'])
    ax2.set_title('predicted segmentation')
    ax2.axis('off')


    # Save as a png image
    f.savefig('../prediction/predict_generatif_plot.jpg')
    # show image
    # plt.show()
    Image.fromarray(mask['Arbre']).save('../prediction/prediction_generatif_{}.jpg'.format(img_name))
