import numpy as np 
import cv2         
 
img_org = cv2.imread(r"C:\Users\USER\Desktop\Department Transfer Review Project\SVD-Based Image Compression with Variable K-Values\Hanni.jpg")    
print('origin image shape is ', img_org.shape)      
img = cv2.resize(img_org, (532, 300))                
print('input image shape is ', img.shape)           

 
def svd_compression(img, k):                                                          
    res_image = np.zeros_like(img)                                                    
    for i in range(img.shape[2]):                                                   
        
        U, Sigma, VT = np.linalg.svd(img[:,:,i])                                   
        res_image[:, :, i] = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])           
 
    return res_image
 
 
res1 = svd_compression(img, k=300)
res2 = svd_compression(img, k=200)
res3 = svd_compression(img, k=100)
res4 = svd_compression(img, k=50)
res = np.vstack((np.hstack((res1, res2)), np.hstack((res3, res4))))
 
cv2.imshow('img', res)  
cv2.waitKey(0)          
cv2.destroyAllWindows() 