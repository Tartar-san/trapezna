import cv2
import imutils

class VideoProcessor:

    def __init__(self, video_path="C:\\Education\\untitled\\videos\\14_04_171019_081736_082929_0129986476.avi"):
        self.stream = cv2.VideoCapture(video_path)

    def show_video(self):
        firstFrame = None
        while True:
            (grabbed, frame) = self.stream.read()

            if not grabbed:
                break

            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            #cv2.imshow("camera", frame)

            if firstFrame is None:
                firstFrame = gray
                continue

            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # loop over the contours
            for c in contours:

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"

            cv2.imshow("Security Feed", frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)

            cv2.waitKey()

    def get_next_frame(self):
        (grabbed, frame) = self.stream.read()


if __name__ == "__main__":
    vidya = VideoProcessor()
    vidya.show_video()