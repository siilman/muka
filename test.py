import cv2
import numpy as np
import os 
import requests
import json
# api_key_value = "9F7j3bA5TAmPt"
# url = "http://inoprex.com/danangskripsi/jembatan.php"

# MASUKIN FILE RECOGNIZER HASIL TRAINING
filePengenal = cv2.face.LBPHFaceRecognizer_create()
filePengenal.read('trainer.yml')
fileHaarcascade = "Cascades/haarcascade_frontalface_default.xml"
haarcasCadeWajah = cv2.CascadeClassifier(fileHaarcascade);
font = cv2.FONT_HERSHEY_SIMPLEX
#COUNTER DIMULAI DARI 0 YA MAS
id = 0
# NAMA SESUAI DENGAN INDEX DI FILE 01
nama = ['zidni', 'nci']

# MULAI KAMERA (BISA DIATUR UKURANNYA SESUAI KEBUTUHAN DAN SPESIFIKASI)
cam = cv2.VideoCapture(0)
cam.set(3, 640) # LEBAR
cam.set(4, 480) # TINGGI
# UKURAN WINDOW
lebarMinimal = 0.1*cam.get(3)
tinggiMinimal = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    wajah = haarcasCadeWajah.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(lebarMinimal), int(tinggiMinimal)),
       )
    for(x,y,w,h) in wajah:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = filePengenal.predict(gray[y:y+h,x:x+w])
        # CEK CONFIDENCE
        if (confidence < 100 and confidence>35):
            id = nama[id]
            confidence = "  {0}%".format(round(100 - confidence))
            # data = "api_key=" + api_key_value + "&nama=" + str(id)+ "&masuk=" + "true" + "&keluar=" + "false" + ""
            # headers = {"Content-Type": "application/x-www-form-urlencoded"}
            # requests.post(url,data=data,headers=headers)
            # response = requests.post(url,headers,data)
            # print(data)
            # print (response)


            #TRIGGER UNTUK KE DATABASE/WEBSITE/KEMANAPUN
            # if (id == "danang") and (confidence >=10):
            # print (id)
            # if (id=="danang"):
            #     print ("HENCEUT")
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
       
        cv2.putText(img, str(id), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) 
        # cv2.putText(img, str("Suhu Tubuh : 36 C"), (x+5,y-5), font, 1, (255,255,255), 2)
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # TEKAN ESC BIAR KELUAR
    if k == 27:
        break
# CLEANUP
print("\n BERSIH BERSIH MASE")
cam.release()
cv2.destroyAllWindows()

