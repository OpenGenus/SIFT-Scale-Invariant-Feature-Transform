import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

class sift_keypoint(torch.nn.modules.Module):
    def __init__(self ,  train_img ,  query_img):
        super(sift_keypoint , self).__init__()
        
        def to_gray(color_img):
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            return gray

        self.train= train_img
        self.query= query_img

    def forward(self ,  draw_n_matches):
        # Initialise SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        train_img_gray = to_gray(self.train)
        query_img_gray = to_gray(self.query)
    
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
            train_img, train_kp,
            query_img, query_kp,
            matches[:N_MATCHES], query_img.copy(), flags=0)
        
        return match_img
