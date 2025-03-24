from PIL import Image
import numpy as np
from collections import Counter

#local imports
class LCProb():
    def __init__(self):
        self.lc_map = {0:'Unknown', 1:'Tree Cover', 2:'Shrubland', 3:'Grassland', 4:'Cropland', 5:'Built-up', 6:'Bare Ground', 7:'Snow/Ice', 8:'Water',
                    9: 'Herbaceous', 10: 'Mangroves', 11: 'Moss and lichen'}
        
        self.lc_pxl_map = { (0,0,0):0,
                            (0,100,0):1,
                           (255, 187, 34):2,
                           (255, 255, 76):3,
                          (240, 150, 255):4,
                          (250, 0, 0):5, 
                         (180, 180, 180):6,
                         (240, 240, 240):7,
                         (0, 100, 200):8, 
                         (0, 150, 160):9,
                         (0, 207, 117):10,
                         (250, 230, 160):11}
        self.lc_pixels = self.lc_pxl_map.keys()
        self.lc_pixels = [np.array(pxl) for pxl in self.lc_pixels]
        
    def discretize_img(self, img):
        h,w,c = img.shape
        new_img = np.zeros_like(img)
        for row in range(h):
            for col in range(w):
                curr_pxl = img[row,col]
                closest_pxl = self.lc_pixels[np.argmin(np.linalg.norm(curr_pxl - self.lc_pixels, axis=-1))]
                new_img[row,col] = closest_pxl
        return new_img 
    
    def im_to_prob(self, img):
        dummy_prob = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}
        h,w,c  = img.shape
        prob_img = np.reshape(img, (-1,3))
        unique_pixels, count = np.unique(prob_img, return_counts=True, axis=0)
        unique_classes = [self.lc_pxl_map[tuple(pxl)] for pxl in unique_pixels]
        class_count = dict(zip(unique_classes, count/(h*w)))
        for key in class_count.keys():
            dummy_prob[key] = class_count[key]
        return dummy_prob

    def prob_to_lc(self, prob):
        d = {}
        for key in prob.keys():
            d[self.lc_map[key]] = prob[key]
        return d

if __name__ == '__main__':
    data_path = '/projects/bdec/adhakal2/hyper_satclip/data/landcover_data/images_corrected/land_cover'
    lc_prob = LCProb()

    new_img = lc_prob.discretize_img(img)
    prob = lc_prob.im_to_prob(new_img)
    lc = lc_prob.prob_to_lc(prob)
    import code; code.interact(local=dict(globals(), **locals()))