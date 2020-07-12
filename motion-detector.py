import cv2
import pandas
from datetime import datetime

# to capture the first frame when webcam starts.
first_frame = None
# list to store status values for each frame.
status_list = [None, None]
# to store timestamp when status values change.
times = []
# to store timestamps in csv file.
df = pandas.DataFrame(columns=["Start Time", "End Time"])

# triggers Video Capture object. 0, 1, 2... for camera, else file path
video = cv2.VideoCapture(0)

while True:
    # boolean and numpy array to read images captured by video
    check, frame = video.read()

    # to indicate whether or not there is object entering/leaving (motion) the current frame
    status = 0

    # convert color frame into grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur image to remove noise & increase accuracy in calculation of diff in delta frame later.
    # first arg is grayscale image, second is width & height of Gaussian kernel(parameters of blurriness), third is std deviation.
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # store the grayscale image of the first frame, to calculate delta frame later.
    if first_frame is None:
        first_frame = gray
        continue

    # calculate diff between two blurried images - first frame and current frame.
    delta_frame = cv2.absdiff(first_frame, gray)

    # set threshold val = 30. check if pixel value is > or < threshold.
    # less indicates no/less motion and is assigned black. vice-versa for white.
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    # smoothen the white areas in threshold image.
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # find all countours/outline of distinct objects in image.
    (cnts, _) = cv2.findContours(thresh_frame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter out contour areas < 10000 (about 100x100 pixels). if contour area >= 10000, draw rect around it.
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        # variable changed to 1 when object greater than threshold is detected.
        status = 1
        # get x, y co-ords, width & height of contour area.
        (x, y, w, h) = cv2.boundingRect(contour)
        # draw green rect around object.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # append current status to the list.
    status_list.append(status)

    # appends current timestamp if current status and previous status are different (i.e object entered/left).
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # recursively display frames in a window.
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    # waits to press key to close window.
    key = cv2.waitKey(1)

    # press 'q' to quit.
    if key == ord('q'):
        # append current timestamp if the object is still in the frame at the time of quitting.
        if status == 1:
            times.append(datetime.now())
        break

# add start and time for each motion, i.e. times[0] indicates when object entered, times[1] indicates when object left and so on.
for i in range(0, len(times), 2):
    df = df.append({"Start Time": times[i], "End Time": times[i+1]}, ignore_index=True)

# save dataframe as csv file.
df.to_csv("Times.csv")

# release webcam.
video.release()

# all windows are destroyed.
cv2.destroyAllWindows
