import cv2
import numpy as np

from utils import parse_datetime
from config import *


class VideoProcessor:

    def __init__(self, video_path="peopleCounter.avi"):
        """
        Subtracting background from the video frames and find contours on the original frame
        which could be a person in the queue.
        :string video_path: path to the video to process
        """
        self.stream = cv2.VideoCapture(video_path)
        self.video_time = parse_datetime(video_path)
        self.background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history = 1000)
        self.min_area = min_contour_area_to_be_a_person
        self.max_area = max_contour_area_to_be_a_person

    @staticmethod
    def crop_interesting_region(frame):
        return frame[interesting_region_y_1:interesting_region_y_2,
               interesting_region_x_1:interesting_region_x_2]

    @staticmethod
    def leave_only_persons(frame, boxes):
        """
        Makes all areas without possible people black
        :ndarray frame: original frame from the video
        :list of (x,y,w,h) boxes: boxes suspected to have a person in
        :ndarray: frame_with_persons: frame with only suspected boxes to have person in
        """
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        for x,y,w,h in boxes:
            mask[y:y+h, x:x+w] = 1

        frame_with_persons = cv2.bitwise_and(frame, frame, mask=mask)
        return frame_with_persons

    def process_frame(self, frame, preview=False, crop=False):
        """
        Subtract background from the video. Leaves only contours which can be a person
        :ndarray frame: frame of the video
        :boolean preview:
        :ndarray: processed_frame: frame with only possible persons in the queue
        """
        if crop:
            frame = self.crop_interesting_region(frame)

        original_frame = frame.copy()

        fg_mask = self.background_subtractor.apply(frame)
        blur = cv2.medianBlur(fg_mask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)  # fill any small holes
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # remove noise

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        good_boxes = []
        # loop over the contours
        for c in contours:
            if cv2.contourArea(c) > self.min_area and cv2.contourArea(c) < self.max_area:
                (x, y, w, h) = cv2.boundingRect(c)
                good_boxes.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        processed_frame = self.leave_only_persons(original_frame, good_boxes)

        if preview:
            cv2.imshow("Frame with boxes", frame)
            cv2.imshow('Foreground', opening)
            cv2.imshow('Only persons', processed_frame)

            cv2.waitKey()

        return processed_frame

    def get_next_frame(self):
        grabbed, frame = self.stream.read()
        return grabbed, frame

    def show_video(self):
        grabbed, frame = self.get_next_frame()

        while grabbed:
            self.process_frame(frame, preview=True)
            grabbed, frame = self.get_next_frame()


if __name__ == "__main__":
    vidya = VideoProcessor()
    vidya.show_video()