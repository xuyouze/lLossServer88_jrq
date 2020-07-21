import os
import cv2

root="/media/data1/xuyouze/LFW"
source = "lfw-deepfunneled"
save_pth="16_lfwa_lr"
# filename = "000001.jpg"
for name in os.listdir(os.path.join(root, source)):
    #print(filename)
    imgpth=os.path.join(root,source,name)
    save_name=os.path.join(root,save_pth,name)
    if not os.path.exists(save_name):
        os.mkdir(save_name)
    for filename in os.listdir(imgpth):
        img=cv2.imread(os.path.join(imgpth,filename))
        w=img.shape[1]
        h=img.shape[0]
        tmp=cv2.resize(img,(16,16),interpolation=cv2.INTER_CUBIC)
        lrimg=cv2.resize(tmp,(w,h))
        # os.mkdir(save_pth)
        cv2.imwrite(os.path.join(save_name,filename),lrimg)
    print(name+".............done saving!\n")
    

""" img=cv2.imread("000001.jpg")
w=img.shape[1]  #178
h=img.shape[0]  #218
 
#放大,双立方插值
newimg1=cv2.resize(img,(int(w/25),int(h/25)),interpolation=cv2.INTER_CUBIC)
newimg2=cv2.resize(newimg1,(w,h),interpolation=cv2.INTER_CUBIC)

cv2.imwrite( "img_bicubic_125f.png", newimg1)
cv2.imwrite( "img_lr_0.125.png", newimg2)

print("Finish") """