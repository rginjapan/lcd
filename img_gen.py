import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(5)

def gasuss_noise(image1, mean=0, var=0.001):
    '''
        添加高斯噪声
        image:原始图像
        mean : 均值
        var : 方差,越大，噪声越大
    '''
    image = image1
    image = np.array(image/255, dtype=float)#将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var ** 0.5, image.shape)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = image + noise#将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)#clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out*255)#解除归一化，乘以255将加噪后的图像的像素值恢复
    #cv.imshow("gasuss", out)
    noise = noise*255
    return [noise,out]

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    # print(denom)
    cos = num / denom
    return cos

def randomization(n):
   x = np.random.random([n,1])
   return x

def L1_norm(vec1,vec2):
    l1=np.linalg.norm((vec1 - vec2), ord=2)
    return l1

def process(visualize=False):

    # img_tmp = np.zeros(480 * 640 * 3)
    # img_a = np.random.multivariate_normal(np.zeros_like(img_tmp), np.eye(len(img_tmp)))
    # img_a=randomization(3)
    img_a = np.random.rand(480,640,3)
    # noise,img_a=gasuss_noise(img_a,mean=0,var=0.03)
    # cv2.imshow('noise', noise)
    # cv2.imshow('out', img_a)
    # cv2.waitKey(0)
    # print(img_a.shape)
    # print(img_a)
    img_b = np.random.rand(480,640,3)
    # print(img_b.dtype)
    # img_b = np.random.multivariate_normal(np.zeros_like(img_tmp), np.eye(len(img_tmp)))
    # img_b = img_a.copy()
    # img_b = randomization(3)
    #print(np.mean(img_a),np.max(img_a), np.min(img_a))

    v_ao = img_a.flatten()
    v_bo = img_b.flatten()
    ss_ori= cos_sim(v_ao, v_bo)
    #img_a /= img_a.max()
    #img_b /= img_b.max()
    circle_pos_a = (np.random.randint(20,300),np.random.randint(10,400))
    circle_rad_a = np.random.randint(10,300)
    # circle_rad_a=170
    cv2.circle(img_a, circle_pos_a, circle_rad_a, (np.random.randn(),np.random.randn(),np.random.randn()), -1)

    circle_pos_b = (np.random.randint(300,600),np.random.randint(10,400))
    # circle_rad_b = 170
    circle_rad_b = np.random.randint(10,300)
    cv2.circle(img_b, circle_pos_b, circle_rad_b, (np.random.rand(),np.random.rand(),np.random.rand()), -1)

    # apply mask
    img_a_new = img_a.copy()
    img_b_new = img_b.copy()
    cv2.circle(img_a_new, circle_pos_a, circle_rad_a, (0,0,0), -1)
    cv2.circle(img_a_new, circle_pos_b, circle_rad_b, (0,0,0), -1)
    cv2.circle(img_b_new, circle_pos_a, circle_rad_a, (0,0,0), -1)
    cv2.circle(img_b_new, circle_pos_b, circle_rad_b, (0,0,0), -1)

    # get score
    v_a = img_a.flatten()
    v_b = img_b.flatten()
    ss = cos_sim(v_a, v_b)
    # ss=v_a-v_b
    # plt.plot(ss)
    # plt.show()
    # ss=cos_sim(v_a,v_b)
    v_a = img_a_new.flatten()
    v_b = img_b_new.flatten()
    ss_new = cos_sim(v_a, v_b)
    # ss_new=cos_sim(v_a,v_b)

    # visualize
    if visualize:
        # print("Similarity score: ", ss, ss_new)
        fig=plt.figure()
        plt.suptitle('Situation 1')
        fig.add_subplot(1,2,1)
        plt.imshow(img_a)
        plt.title("image A")
        fig.add_subplot(1,2,2)
        plt.title("image B")
        plt.imshow(img_b)
        # fig.add_subplot(2,2,3)
        # plt.title("image A with mask")
        # plt.imshow(img_a_new)
        # fig.add_subplot(2,2,4)
        # plt.title("image B with mask")
        # plt.imshow(img_b_new)
        plt.tight_layout()
        plt.show()


    return ss_ori, ss , ss_new, img_b
    #print("Similarity score (original)", ss)
    #print("Similarity score (post)", ss_new)

if __name__ == '__main__':

    ss_ = []
    ss_new_ = []
    rel_ = []
    per_=[]
    ss_ori_=[]
    for i in range(1000):
        ss_ori, ss, ss_new, img = process(False)
        ss_.append(ss)
        ss_new_.append(ss_new)
        ss_ori_.append(ss_ori)
        #
        rel = (ss_new-ss)#/ss
        rel_.append(rel)
        # print(ss_new)
        per=rel/ss
        per_.append(per)
    # print(np.mean(rel_))
    # print(np.mean(ss_))
    # print(np.mean(ss_new_))
    # print("per cent",np.mean(rel_)/np.mean(ss_) )
    # print("per cent = ", np.mean(per_))
        # if rel>2:
        #     cv2.imshow('win', img)
        #     cv2.waitKey(0)
    plt.title('cosine similarity score on situation 2.2')  # give plot a title
    plt.xlabel('Id of image pairs')  # make axis labels
    plt.ylabel('cosine distance')
    plt.plot(ss_, label='original')
    # plt.plot(ss_ori_, label='original without occluded circles')
    plt.plot(ss_new_, label='score with mask')
    # plt.plot(rel_, label='difference')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
