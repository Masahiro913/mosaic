#opencvの顔検出は横顔や傾きに弱く検出できない
#これを確認するためにgirl.pngを回転させてどこまで検出できるかをテストする
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

#顔検出と画像の読み込み
cascade_file = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)
img = cv2.imread("girl.jpg")

#顔検出を実行し、印をつける
def face_detect(img):
  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  face_list = cascade.detectMultiScale(img_gray,minSize=(300,300))
  #認識した箇所に印をつける
  for (x,y,w,h) in face_list:
    print("顔の座標=", x,y,w,h)
    red = (0,0,255)
    cv2.rectangle(img,(x,y),(x+w,y+h),red,thickness=30)

#角度毎に検証する
for i in range(0,9):
  ang = i * 10
  print("---" + str(ang) + "---")
  img_r = ndimage.rotate(img, ang)
  face_detect(img_r)
  plt.subplot(3,3,i+1)
  plt.axis("off")
  plt.title("angle="+ str(ang))
  plt.imshow(cv2.cvtColor(img_r,cv2.COLOR_BGR2RGB))

plt.show()

#赤枠が現れない。。。