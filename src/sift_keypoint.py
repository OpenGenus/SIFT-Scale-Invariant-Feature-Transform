import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms



class sift_keypoint(torch.nn.modules.Module):
    def __init__(self ,  train_img ,  query_img):
        super(sift_keypoint , self).__init__()
        
        def to_gray(color_img):
            gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
            return gray

        self.train= train_img.permute(1,2,0)
        self.query= query_img.permute(1,2,0)
        self.to_gray = to_gray
        

    def forward(self ,  draw_n_matches):
        # Initialise SIFT detector
        sift = cv2.SIFT_create()
        
        to_gray = self.to_gray
        train_image = self.train
        query_image = self.query
        
        train_img_gray = to_gray(train_image.numpy())
        query_img_gray = to_gray(query_image.numpy())

        
        # Generate SIFT keypoints and descriptors
        train_kp, train_desc = sift.detectAndCompute(train_img_gray, None)
        query_kp, query_desc = sift.detectAndCompute(query_img_gray, None)

        # create a BFMatcher object which will match up the SIFT features
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = bf.match(train_desc, query_desc)

        # Sort the matches in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # draw the top N matches
        N_MATCHES = draw_n_matches

        match_img = cv2.drawMatches(
            train_image.numpy(), train_kp,
            query_image.numpy(), query_kp,
            matches[:N_MATCHES], query_image.numpy(), flags=0)
        
        return match_img , matches , train_kp , train_desc , query_kp , query_desc
