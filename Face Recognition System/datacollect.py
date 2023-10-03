import cv2
import os

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# nameID=str(input("Enter Your Name: ")).lower()

# path='images/'+nameID

# isExist = os.path.exists(path)

# if isExist:
# 	print("Name Already Taken")
# 	nameID=str(input("Enter Your Name Again: "))
# else:
# 	os.makedirs(path)

count = 0

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)

    for x, y, w, h in faces:
        newX = x / 640
        newY = y / 480
        newW = w / 640
        newH = h / 480

        print(newX, newY, newW, newH)
		
        name = "./images/face_00" + str(count) + ".jpg"
        
        if count > 9:
            name = "face_0" + str(count) + ".jpg"
        if count > 99:
            name = "face_" + str(count) + ".jpg"
        
        

        cv2.imwrite(name, frame)
        with open("./images/label.txt", "a") as myfile:
            myfile.write(
                name + ", " + str(newX) + ", " + str(newY) + ", " + str(newW) + ", " + str(newH) + "\n"
            )

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        count += 1

    cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1)
    if ord("q") == cv2.waitKey(1):
        break
    if count > 500:
        break
video.release()
cv2.destroyAllWindows()
