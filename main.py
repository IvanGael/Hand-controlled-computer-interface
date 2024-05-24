import cv2
import numpy as np
import pyautogui

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the dimensions of the screen
screen_width, screen_height = pyautogui.size()

# Set up the hand and finger detector
hand_cascade = cv2.CascadeClassifier('haarcascade_hand.xml')   

# Parameters for click and scroll actions
scroll_threshold = 50  
click_threshold = 30  
prev_x, prev_y = 0, 0
scrolling = False

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    # If hands are detected, count the number of fingers
    if len(hands) > 0:
        # Assuming one hand is shown
        (hx, hy, hw, hh) = hands[0]

        # Crop the region of interest around the hand
        hand_roi = gray[hy:hy+hh, hx:hx+hw]

        # Apply thresholding to isolate the hand
        _, hand_thresh = cv2.threshold(hand_roi, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(hand_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of fingers (convexity defects)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull)

            if defects is not None:
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, _, _ = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[defects[i, 0][2]][0])

                    # Calculate the triangle area formed by each defect
                    a = np.linalg.norm(np.array(far) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(end))
                    c = np.linalg.norm(np.array(start) - np.array(end))
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                    # If angle < 90 degrees, it's a finger
                    if angle < np.pi/2:
                        finger_count += 1

                # Perform actions based on finger count
                if finger_count == 1:
                    pyautogui.click()  # Click action with one finger
                elif finger_count == 5:
                    pyautogui.scroll(10)  # Scroll action with all fingers

    # Display the frame
    cv2.imshow('Hand Gesture Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
