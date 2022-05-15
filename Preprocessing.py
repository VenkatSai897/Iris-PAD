import torch
import torch.utils.data as data_utl
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from keras.datasets import cifar10
class datasetLoader(data_utl.Dataset):

    def __init__(self, split_file, root, train_test, random=True, c2i={}):
        self.class_to_id = c2i
        self.id_to_class = []

        # Class assignment
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0

        # Image pre-processing
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(1,sigma = (0.1,2.0)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.Normalize(mean=[0.485],std=[0.229])
            ])

        # Reading data from CSV file
        SegInfo=[]
        x = 0;
        with open(split_file, 'r') as f:
            for l in f.readlines():
                
                v = l.strip().split(',')
               
                
                if train_test == v[0]:
                    image_name = v[1]
                    
                    imagePath = root +image_name
                    
                    
                    c = v[2]
                    
                    if c not in self.class_to_id:
                        self.class_to_id[c] = cid
                        self.id_to_class.append(c)
                        cid += 1
                    # Storing data with imagepath and class
                    self.data.append([imagePath, self.class_to_id[c]])
                   


        self.split_file = split_file
        self.root = root
        self.random = random
        self.train_test = train_test
        

    def __getitem__(self, index):
      imagePath, cls = self.data[index]
      
      imageName = imagePath.split('/')[-1]
      # Reading of the image
      path = imagePath
    
      img = Image.open(path)

      # Applying transformation
      tranform_img = self.transform(img)
      img.close()

      # Repeat NIR single channel thrice before feeding into the network
      tranform_img= tranform_img.repeat(3,1,1)
      return tranform_img[0:3,:,:], cls, imageName
    
    def __len__(self):
      return len(self.data)

if __name__ == '__main__':

    dataseta = datasetLoader('drive/MyDrive/Data Preprocessing/Train_Test_split(csv).csv', 'drive/MyDrive/Data Preprocessing/Training_DataSet/TrainImages/', train_test='test')
    

    


