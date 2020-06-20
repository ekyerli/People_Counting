# import the necessary packages
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from mobilenet_ssd.classes import CLASSES


def counter(filenameOpen, filenameSave):

    # örnek1-2 için 0.55,30;örnek3 için 0.5,5;örnek4 için 0.45,11;
    defaultConfidence = 0.72 #minimum algılama yüzdesi
    defaultSkipFrames = 30 # atlamalar arası atlanan frame
    W = None
    H = None
    writer = None

    # yükleme modeli-yapay sinir ağları
    net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                   "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

    print("Video yükleniyor..")
    vs = cv2.VideoCapture(filenameOpen)

    # nesneleri saklamak için her birine id veriyoruz.
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # şimdiye kadar işlenen toplam kare sayısı
    # yukarı veya aşağı hareket eden toplam nesne sayısı
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # saniye başına kare çıktısı tahmincisi
    fps = FPS().start()
    stat = {}

    while True:
        # VideoCapture
        ok, frame = vs.read()

        # videonun sonu gelir ve döngüden çıkarız.
        if filenameOpen is not None and frame is None:
            break

        # çerçeveyi maksimum 500 piksel olacak şekilde yeniden boyutlandırdık.
        # (ne kadar az veriye sahipsek, o kadar hızlı işleme koyabiliriz),
        # ardından dlib için çerçeveyi BGR'den RGB'ye dönüştürdük.

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # çerçeve boyutları videoya göre ayarlanıyor.
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # dosyayı kaydetmek için
        if filenameSave is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(filenameSave, fourcc, 30, (W, H), True)

        status = "Bekleniyor"
        rects = []

        # işlemciyi yormamak için örnekleme zamanını ayarladık

        videotime = vs.get(cv2.CAP_PROP_POS_MSEC) / 1000
        summ = totalUp + totalDown

        if totalFrames % 50 == 0:
            stat["{:.4s}".format(str(videotime))] = str(summ)
            #giriş çıkış toplam sayıyı yazdırmak için
            #print("{:.4s}".format(str(videotime)) + " people: " + str(summ))

        if totalFrames % defaultSkipFrames == 0:
            # set the status and initialize our new set of object trackers
            status = "Bulundu"
            trackers = []

            # çerçeve boyutlarını bloba dönüştürdüğümüz kısım
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # tespit edilen tüm nesneler
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # tahmin değeri eşik değerinden büyük mü?
                if confidence > defaultConfidence:
                    idx = int(detections[0, 0, i, 1])

                    #insan değilse yoksay
                    if CLASSES[idx] != "person":
                        continue

                    # nesneyi çerçeveye alıyoruz.
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # listtemize ekliyoruz
                    trackers.append(tracker)
                    

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                status = "Takip"

                # izleyiciyi güncelleyin ve güncellenmiş konumu yakalayın
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # sınırlayıcı kutu koordinatlarını dikdörtgenler listesine ekledik
                rects.append((startX, startY, endX, endY))

        # yukarı mı aşağı mı gittiklerini anlamak için bir çizgi çekiyoruz
        #burayı değiştirirsek 169. satırıda düzeltmemiz gerekli.
        #cv2.line(frame, (0, 0), (W, H), (0, 255, 255), 2)#çapraz
        cv2.line(frame, (0, H // 2), (W, H // 2), (255, 255, 0), 2)#yatay
        #cv2.line(frame, (W//2, 0), (W//2, H), (0, 255, 255), 2)  # dikey

        # nesneler ilişkilendiriyoruz.
        objects = ct.update(rects)

        # nesneyi döngüye sokuyoruz
        for (objectID, centroid) in objects.items():
            # geçerli nesne kimliği için izlenebilir bir nesne olup olmadığını kontrol ettik.
            to = trackableObjects.get(objectID, None)

            #yoksa oluşturuyoruz.
            if to is None:
                to = TrackableObject(objectID, centroid)

            # varsa yön belirlemek için bu fonksiyona giriyoruz.
            else:
                # hareket yönüne göre aşağı mı yoksa yukarı mı diye bakıyoruz.
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)


                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True

                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True

            # izlenebilir nesnemize id veriyoruz.
            trackableObjects[objectID] = to

            #nesnenin merkezini seçiyoruz ve id sini yazıyoruz.
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)

        # ekran çıktıları
        info = [
            ("Yukari", totalUp),
            ("Asagi", totalDown),
            ("Sure", "{:.2f}".format(videotime))
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # videoyu kaydetmişyiz diye kontrol ediyoruz.
        if writer is not None:
            writer.write(frame)

        # show the output frame
        cv2.imshow("Ege Universitesi Bitirme Tezi(Caner YILDIRIM-Emir Kaan YERLI)", frame)
        key = cv2.waitKey(1) & 0xFF

        # döngüden çıkış için q,Q bas
        if key == ord("q") or key == ord("Q"):
            break
        # fram güncelliyoruz
        totalFrames += 1
        fps.update()

    # durduruyoruz
    fps.stop()

    # videoyu kaydetmeyi sonlandırıyoruz.
    if writer is not None:
        writer.release()

    # if we are not using a video file, stop the camera video stream
    if not filenameOpen:
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()

    # pencereleri kapa
    cv2.destroyAllWindows()

    #print(stat)

    return info, stat
# if __name__ == "__main__":
#     counter()
