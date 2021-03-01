import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(5)

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    # print(denom)
    cos = num / denom
    return cos

pool1=240*320*64
pool2=120*160*128
pool3=60*80*256
pool4=30*40*512
pool5=15*20*512
fc6=4096

def process():
    vec_a = np.random.rand(1,pool1)
    vec_b = np.random.rand(1,pool1)
    ss = cos_sim(vec_a,vec_b)





    return ss





if __name__ == '__main__':
    ss_=[]
    for i in range(1000):
        ss = process()
        ss_.append(ss)



    plt.title('cosine similarity score on situation 1.3')  # give plot a title
    plt.xlabel('Id of image pairs')  # make axis labels
    plt.ylabel('cosine distance')
    plt.plot(ss_, label='original with occluded circles')
    # plt.plot(ss_ori_, label='original without occluded circles')
    # plt.plot(ss_new_, label='score with mask')
    # plt.plot(rel_, label='difference')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

