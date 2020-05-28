import cv2
import os
import numpy as np


#function to detect the existence of a face
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img , cv2.COLOR_BGR2GRAY)
    face_haar= cv2.CascadeClassifier(r"C:\Users\User\Desktop\DATA SCIENCE\face recognition using LBPH\haarcascade_frontalface_alt.xml")
    faces = face_haar.detectMultiScale(gray_img,scaleFactor = 1.3,minNeighbors=3)
    return faces,gray_img

#function to label the training data
def labeling_training_data(directory):
    faces=[]
    faceID=[]
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("not a proper file")
                continue
            my_id = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print("image_id:",my_id)
            print("img_path:",img_path)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("not loaded proerly")
                continue
            faces_rect , gray_img = faceDetection(test_img)
            if len(faces_rect)!=1:
                continue
            (x,y,w,h) = faces_rect[0]
            cropped_image = gray_img[y:y+w,x:x+h]
            faces.append(cropped_image)
            faceID.append(my_id)
    return faces,faceID

#function to recognize the face (pairing the testing video/picture with the true label)
def faceRecognizer(faces,faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

#function to draw a rectangle bounding the face after detecting the existence of a face
def drawRectangle(test_img,face):
    faces,gray_img = faceDetection(test_img)
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

#function to type a text (name of the predicted person)
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,0),6)