import cv2
import sys
import math
file=sys.argv[1]
last_num=sys.argv[2]
i=0
for i in range(int(last_num)):
    img=cv2.imread(file+str(i)+".jpg")
    init_x=80
    init_y=100
    pt1=(init_x,init_y)
    l=50
    angle=i
    rad=math.radians(angle)
    x=int(l*math.cos(rad))
    y=int(l*math.sin(rad))

    pt2=(init_x+x,init_y-y)

    color=(0,0,0)
    thickness=2
    lineType=8
    shift=0
    cv2.line(img,pt1,pt2,color,thickness,lineType,shift)
    vx = pt2[0] - pt1[0]
    vy = pt2[1] - pt1[1]
    v  = math.sqrt(vx ** 2 + vy ** 2)
    ux = vx / v
    uy = vy / v

    w = 5
    h = 10
    ptl = (int(pt2[0] - uy*w - ux*h), int(pt2[1] + ux*w - uy*h))
    ptr = (int(pt2[0] + uy*w - ux*h), int(pt2[1] - ux*w - uy*h))

    cv2.line(img,pt2,ptl,color,thickness,lineType,shift)
    cv2.line(img,pt2,ptr,color,thickness,lineType,shift)
    cv2.imwrite(file+str(i)+".jpg",img)
#cv2.imshow("result",img)
#cv2.waitKey(0)