import sys

vidStream = cv2.VideoCapture(0)

i=0
while True:
    _,frame= vidStream.read()  #read frame and return code
    cv2.imshow("testing window",frame)
    cv2.imwrite("C:/Users/User/Face recognition using LBPH/camera dataset/0/image"+str(i)+".jpg",frame)
    i+=1
    if cv2.waitKey(10)==ord("q"):
        break