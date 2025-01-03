import cv2 as cv

model_bin = "D:/projects/opencv_tutorial_data/models/face_detector/opencv_face_detector_uint8.pb"
config_text = "D:/projects/opencv_tutorial_data/models/face_detector/opencv_face_detector.pbtxt";


def video_detection():
    # load tensorflow model
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    capture = cv.VideoCapture("D:/images/video/example_dsh.mp4")

    # 人脸检测
    while True:
        e1 = cv.getTickCount()
        ret, frame = capture.read()
        if ret is not True:
            break
        h, w, c = frame.shape
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False);
        net.setInput(blobImage)
        cvOut = net.forward()

        # Put efficiency information.
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        # 绘制检测矩形
        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            objIndex = int(detection[1])
            if score > 0.5:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h

                # 绘制
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                cv.putText(frame, "score:%.2f"%score, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        e2 = cv.getTickCount()
        fps = cv.getTickFrequency() / (e2 - e1)
        cv.putText(frame, label + (" FPS: %.2f"%fps), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv.imshow('face-detection-demo', frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_detection()

