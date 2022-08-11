import cv2
from tracker2 import *
import numpy as np
end = 0

#create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("Main_video.mp4")
f = 25
w = int(1000/(f-1))
print(w)

#object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=None, varThreshold=None)
#100,5

#Kernels

kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernel_e = np.ones((5,5),np.uint8)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,None,fx = 0.5, fy = 0.5)
    height,width,_ = frame.shape

    #extract roi
    roi = frame[50:540, 200:960]

    #masking method
    mask = object_detector.apply(roi)
    mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

    #different masking method
    fgmask = fgbg.apply(roi)
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN , kernelOp)
    mask2 = cv2.morphologyEx(mask1 , cv2.MORPH_CLOSE,kernelCl)
    e_img = cv2.erode(mask2, kernel_e)

    contours,_ = cv2.findContours(e_img , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #threshold
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w,y+h) , (0,255,0) , 3)
            detections.append([x,y,w,h])
    #object tracking
    boxes_id = tracker.update(detections)
    for box_id in boxes_id:
        x,y,w,h,id = box_id


        if(tracker.getsp(id) < tracker.limit()):
            cv2.putText(roi , str(id) + " " + str(tracker.getsp(id)), (x,y-15) , cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 2)
            cv2.rectangle(roi , (x,y) , (x+w,y+h) , (0,255,0) , 3)
        else:
            cv2.putText(roi , str(id) + " " + str(tracker.getsp(id)) , (x, y-15), cv2.FONT_HERSHEY_PLAIN , 1 , (0,0,255) ,2)
            cv2.rectangle(roi, (x,y) , (x+w,y+h) , (0,165,255) , 3)

        s = tracker.getsp(id)
        if (tracker.f[id] == 1 and s!= 0 ):
            tracker.capture(roi , x , y , h , w  ,s , id)

        
    #draw lines

    cv2.line(roi, (0, 410), (960, 410), (0, 0, 255), 2)
    cv2.line(roi, (0, 430), (960, 430), (0, 0, 255), 2)
    

    cv2.line(roi, (0, 235), (960, 232), (0,0,255), 2)
    cv2.line(roi, (0, 255), (960, 255), (0,0,255), 2)
    
    #display
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(w-10)
    if key == 27:
        tracker.end()
        end = 1
        break
    if cv2.waitKey(3) & 0xFF == ord('w'):
        break


    def end(self):
            file = open("SpeedRecord.txt", "w")
            file.write("\n-------------\n")
            file.write("-------------\n")
            file.write("SUMMARY\n")
            file.write("-------------\n")
            file.write("Total Vehicles :\t" + str(self.count) + "\n")
            file.write("Exceeded speed limit :\t" + str(self.exceeded))
            file.close()


    # click the picture of all the vehicles crossing the line
    def capture(self, img, x, y, h, w, sp, id):
        if(self.capf[id] == 0):
            self.capf[id] = 1
            self.f[id] = 0
            crop_img = img[y-5:y + h+5, x-5:x + w+5]
            n = str(id)+"_speed_"+str(sp)
            file = 'D://TrafficRecord//SpeedRecord.txt' + n + '.jpg'
            cv2.imwrite(file, crop_img)
            self.count += 1
            filet = open("SpeedRecord.txt", "a")
            if( sp > limit ):
                file2 = 'D://TrafficRecord//SpeedRecord.txt' + n + '.jpg'
                cv2.imwrite(file2, crop_img)
                filet.write(str(id)+" \t "+str(sp)+"<---exceeded\n")
                self.exceeded += 1
            else:
                filet.write(str(id) + " \t " + str(sp) + "\n")
            filet.close()



if(end!=1):
    tracker.end()

cap.release()
cv2.destroyAllWindows()

