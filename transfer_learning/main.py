from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import cv2
import loss
from model import Vgg

color = (67,67,67)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def processimg(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

model = Vgg()

pictures = "Test_samples/"
Data = dict()

for file in listdir(pictures):
	picture, ext = file.split(".")
	Data[picture] = model.predict(processimg("Test_samples/%s.jpg" % (picture)))[0,:]

###### IMAGE VERFICATION ##########

def VerifyFace(img1, folder):
	epsilon = 0.4
	for file in listdir(folder):
		img1_representation = model.predict(processimg(img1))[0,:]
		img2_representation = model.predict(processimg(folder + file))[0,:]
 
		cosine_similarity = loss.cosine_similarity(img1_representation, img2_representation)
		euclidean_distance = loss.EuclideanDistance(img1_representation, img2_representation)
 
		if(cosine_similarity < epsilon):
			print(img1 + " " + "verified... they are same person" + " " + file)
		else: 
  			print(img1 + " " + "unverified! they are not same person!" + " " + file )

VerifyFace("Test_samples/Scarlett.jpg", pictures)

####### Video Testing #################

cap = cv2.VideoCapture(0) # 0,1,2 if multiple webcams

while(True):
	ret, img = cap.read()
	#img = cv2.resize(img, (640, 360))
	faces = face_cascade.detectMultiScale(img, 1.3, 3)
	
	for (x,y,w,h) in faces:
		if w > 130: 
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 127.5
			img_pixels -= 1
			
			captured = model.predict(img_pixels)[0,:]
			
			found = 0
			for i in Data:
				picture = i
				test = Data[i]
				
				similarity = loss.cosine_similarity(test, captured) 	 #EuclideanDistance  similarity < 0.4
									 		# cosine_similarity  similarity < 120
				#print(similarity)
				if(similarity < 0.3):
					cv2.putText(img, picture, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
					
					found = 1
					break
					
		
			if(found == 0):
				cv2.putText(img, 'unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
	
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()

	

