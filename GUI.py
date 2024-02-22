import os
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image #pip install pillow
import numpy as np


def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor = 1.3
        # minimum neighbor = 5
         
        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
     
    cap = cv2.VideoCapture(0)
    id = 3
    img_id = 0
     
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "data/thief."+str(id)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
             
            cv2.imshow("Cropped face", face)
             
        if cv2.waitKey(1)==13 or int(img_id)==300: #13 is the ASCII character of Enter
            break
             
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed....")


def train_classifier():
    data_dir = "D:/VISAI/final2/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
     
    faces = []
    ids = []
     
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
         
        faces.append(imageNp)
        ids.append(id)
         
    ids = np.array(ids)
     
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")


def prediction():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
        
        for (x,y,w,h) in features:
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
            
            id, pred = clf.predict(gray_img[y:y+h,x:x+w])
            confidence = int(100*(1-pred/300))
            
            if confidence>75:
                if id==1:
                    cv2.putText(img, "Noble", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
                if id==2:
                    cv2.putText(img, "Dino", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
                if id==3:
                    cv2.putText(img, "Pori", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN CIVILIAN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)
        
        return img
    
    # loading classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, img = video_capture.read()
        img = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
        cv2.imshow("face Detection", img)
        
        if cv2.waitKey(1)==13:
            break
    video_capture.release()
    cv2.destroyAllWindows()


########################################################
 
window = tk.Tk()
window.title("Face Recognition system")
 
l1 = tk.Label(window, text="Name", font=("Algerian",20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)
 
l2 = tk.Label(window, text="Age", font=("Algerian",20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)
 
l3 = tk.Label(window, text="Address", font=("Algerian",20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)
 
 
b1 = tk.Button(window, text="Training", font=("Algerian",20),bg="orange",fg="red",command=train_classifier)
b1.grid(column=0, row=4)
 
b2 = tk.Button(window, text="Detect the faces", font=("Algerian",20), bg="green", fg="orange",command=prediction)
b2.grid(column=1, row=4)
 
b3 = tk.Button(window, text="Generate dataset", font=("Algerian",20), bg="pink", fg="black",command=generate_dataset)
b3.grid(column=2, row=4)
 
window.geometry("800x200")
window.mainloop()