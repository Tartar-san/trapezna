import cv2

class PeopleDetector:

    def __init__(self, background_image="assets/background.jpg"):
        self.background_image = cv2.imread(background_image)

    def detect_people(self, img):
        """

        :return: list of boundary boxes
        """
        pass


if __name__ == "__main__":
    pass