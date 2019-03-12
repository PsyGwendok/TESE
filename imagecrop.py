from PIL import Image
import cv2
img = cv2.imread("C:/Users/Psy/Downloads/Data/Leopards/emammal_image_d16750s4i1.jpg")

crop_img=img[422:618,459:970 ]

cv2.imwrite("temp.png",crop_img)

cv2.imshow("img",crop_img)
cv2.waitKey()