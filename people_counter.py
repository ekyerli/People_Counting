#main processing file
from mobilenet_ssd.classes import CLASSES
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import dlib
import cv2



def counter(filenameOpen, filenameSave):
    # 0.55,30 for örnek1-2  ; 0.5,5 for örnek3 ;0.45,11 for örnek4 ;0,55,20 for örnek5 ;
    defaultConfidence = 0.55 #threshold value
    defaultSkipFrames = 20 #skipped frame
    W = None
    H = None
    writer = None

    # loading model-artificial neural networks
    net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                   "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

    print("Video yükleniyor..")
    vs = cv2.VideoCapture(filenameOpen)

    #each id is given to store objects.25,20 for örnek 6; others 40,50 ;
    ct = CentroidTracker(maxDisappeared=25, maxDistance=20)#boundary lines of objects are determined
    trackers = []
    trackableObjects = {}

    # variables are determined
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    #fps output estimator per second
    fps = FPS().start()
    stat = {}

    while True:
        # VideoCapture
        ok, frame = vs.read()


        if filenameOpen is not None and frame is None:
            break

        #we resized the frame to a maximum of 500 pixels.
        #(the less data it has, the faster we can process it),
        #then convert the frame for dlib from BGR to RGB.

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #frame sizes are adjusted according to the video.
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        #to save the file
        if filenameSave is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(filenameSave, fourcc, 30, (W, H), True)

        status = "Bekleniyor"
        rects = []

        #we set the sampling time to avoid tiring the processor
        videotime = vs.get(cv2.CAP_PROP_POS_MSEC) / 1000
        summ = totalUp + totalDown

        if totalFrames % 50 == 0:
            stat["{:.4s}".format(str(videotime))] = str(summ)
            #input to print total number
            #print("{:.4s}".format(str(videotime)) + " people: " + str(summ))

        if totalFrames % defaultSkipFrames == 0:
            # set the status and initialize our new set of object trackers
            status = "Bulundu"
            trackers = []

            #the part where we convert the frame dimensions to blob
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            #all detected objects
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                #is the forecast value greater than the threshold value?
                if confidence > defaultConfidence:
                    idx = int(detections[0, 0, i, 1])

                    #ignore if not human
                    if CLASSES[idx] != "person":
                        continue

                    #frame the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

        else:
            #loop over the trackers
            for tracker in trackers:
                status = "Takip"

                #update the viewer and capture the updated location
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                #bounding box coordinates added to list of rectangles
                rects.append((startX, startY, endX, endY))

        #A line is drawn to see if they go up or down
        #If it changes here, line 149 needs to be corrected.
        #cv2.line(frame, (0, 0), (W, H), (0, 255, 255), 2)#cross
        cv2.line(frame, (0,H // 2), (W, H// 2), (255, 255, 0), 2)#horizontal
        #cv2.line(frame, (W//2, 0), (W//2, H), (0, 255, 255), 2)  #vertical

        #associating objects.
        objects = ct.update(rects)

        #object is looped
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)

            #this function is entered to determine the direction.
            else:
                #depending on the direction of movement, whether it is up or down
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < (H) // 2:
                        totalUp += 1
                        to.counted = True

                    elif direction > 0 and centroid[1] > (H) // 2:
                        totalDown += 1
                        to.counted = True

            #traceable object is given id
            trackableObjects[objectID] = to

            #the center of the object is selected and the id number is written
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)

        #screen printouts
        info = [
            ("Yukari", totalUp),
            ("Asagi", totalDown),
            ("Sure", "{:.2f}".format(videotime))
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        #checking if the video has been saved
        if writer is not None:
            writer.write(frame)

        # show the output frame
        cv2.imshow("Ege Universitesi Bitirme Tezi(Caner YILDIRIM-Emir Kaan YERLI)", frame)
        key = cv2.waitKey(1) & 0xFF

        #exit the loop
        if key == ord("q") or key == ord("Q"):
            break
        totalFrames += 1
        fps.update()

    fps.stop()

    #video recording is ending
    if writer is not None:
        writer.release()

    #if we are not using a video file, stop the camera video stream
    if not filenameOpen:
        vs.stop()

    #otherwise, release the video file pointer
    else:
        vs.release()

    #close the windows
    cv2.destroyAllWindows()

    return info, stat
# if __name__ == "__main__":
#     counter()
