import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

class customDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform
        self.volA = os.listdir(root_A)
        self.volB = os.listdir(root_B)

        #on ne va pas utiliser toutes les images, on va prendre le minimum des 2 datasets
        self.length_dataset = min(len(self.volA), len(self.volB))
        self.volA_len = len(self.volA)
        self.volB_len = len(self.volB)


        self.slicesA = []
        self.slicesB = []

        #extraction des slices de chaque volume dans 2 listes
        for i in range(self.length_dataset):
            volA_path = os.path.join(self.root_A, self.volA[i])
            volB_path = os.path.join(self.root_B, self.volB[i])

            volA = nib.load(volA_path)
            volB = nib.load(volB_path)

            volA = volA.get_fdata()
            volB = volB.get_fdata()

            for j in range(volA.shape[0]): #on parcourt les slices du volume
                self.slicesA.append(volA[j,:,:]) #on ajoute la slice à la liste
            
            for j in range(volB.shape[0]): #on parcourt les slices du volume
                self.slicesB.append(volB[j,:,:]) #on ajoute la slice à la liste

        
        self.length_dataset = min(len(self.slicesA), len(self.slicesB))

        #on cherche la dimension minimale en x et en y pour pouvoir les mettre à la même taille
        self.min_x = 100000
        self.min_y = 100000
        for i in range(self.length_dataset):
            if self.slicesA[i].shape[0] < self.min_x:
                self.min_x = self.slicesA[i].shape[0]
            if self.slicesA[i].shape[1] < self.min_y:
                self.min_y = self.slicesA[i].shape[1]
            
            if self.slicesB[i].shape[0] < self.min_x:
                self.min_x = self.slicesB[i].shape[0]
            if self.slicesB[i].shape[1] < self.min_y:
                self.min_y = self.slicesB[i].shape[1]
            
        #on met à la même taille les slices
        for i in range(self.length_dataset):
            self.slicesA[i] = self.slicesA[i][0:self.min_x, 0:self.min_y]
            self.slicesB[i] = self.slicesB[i][0:self.min_x, 0:self.min_y]

        #on normalise les slices
        for i in range(self.length_dataset):
            self.slicesA[i] = self.slicesA[i] / np.max(self.slicesA[i])
            self.slicesB[i] = self.slicesB[i] / np.max(self.slicesB[i])
        

        #on transforme les slices en float
        for i in range(self.length_dataset):
            self.slicesA[i] = self.slicesA[i].astype(np.float32)
            self.slicesB[i] = self.slicesB[i].astype(np.float32)
        


    def __len__(self):
        return self.length_dataset 

    def __getitem__(self, index):
        return self.slicesA[index], self.slicesB[index] #on renvoie les slices de chaque volume
        