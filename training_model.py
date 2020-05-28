import Face_Recognition as fr

#training the model on the  training dataset created using both a camera
faces,faceID = fr.labeling_training_data(r"C:\Users\User\Face recognition using LBPH\camera dataset")

faceID = list(map(int, faceID))
face_recognizer = fr.faceRecognizer(faces,faceID)
name = {0:"Hadi",1:"Jon Snow"}

#saving the model
face_recognizer.save(r"C:\Users\User\Face recognition using LBPH\face_recog.yml")