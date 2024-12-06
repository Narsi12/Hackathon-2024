import cv2
import os
import cvzone
from cvzone.PoseModule import PoseDetector
 
# Initialize video capture
cap = cv2.VideoCapture(0)
 
# Initialize pose detector
detector = PoseDetector(detectionCon=0.5, trackCon=0.5)
 
# Load left button image
imageButtonLeftPath = r"D:\Gen-AI-2024\Resources\button 1.png"
imageButtonLeft = cv2.imread(imageButtonLeftPath, cv2.IMREAD_UNCHANGED)
 
# Check if the button image is loaded successfully
if imageButtonLeft is None:
    print(f"Error: Failed to load the button image from {imageButtonLeftPath}")
    exit()
else:
    print(f"Left button image loaded successfully from {imageButtonLeftPath}")
 
# Convert button image to 4 channels if not already
if imageButtonLeft.shape[2] == 3:  # Check if the image has only 3 channels
    imageButtonLeft = cv2.cvtColor(imageButtonLeft, cv2.COLOR_BGR2BGRA)
 
# Load T-shirt images from resources
TshirtFolderPath = r"D:\Gen-AI-2024\Resources\resources"
listImages = os.listdir(TshirtFolderPath)
imageIndex = 0  # Current T-shirt image index
 
# Fixed ratio for T-shirt dimensions
fixedRatio = 260 / 190
scale_factor = 1.5  # Scaling factor for T-shirt size
offset_y = 50  # Vertical offset to position T-shirt
 
# Initialize counter for button gesture
counterLeft = 0
 
while True:
    # Read frame
    success, img = cap.read()
    if not success:
        print("Failed to read frame or end of video.")
        break
 
    # Detect pose landmarks
    img = detector.findPose(img, draw=True)
    lmlist, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
 
    if len(lmlist) >= 25:  # Ensure all keypoints are detected
        # Draw the left button
        img = cvzone.overlayPNG(img, imageButtonLeft, (72, 293))   # Left button
 
        # Right wrist detection for left button gesture
        lm16 = lmlist[16]  # Right wrist
        if lm16[0] < 150 and lm16[0] > 72:  # Left button range
            counterLeft += 1
            if counterLeft > 10:  # Trigger gesture after a threshold
                counterLeft = 0
                # Decrement T-shirt image index
                imageIndex = (imageIndex - 1) % len(listImages)
        else:
            counterLeft = 0
 
        # Load current T-shirt image
        tshirtImg = cv2.imread(os.path.join(TshirtFolderPath, listImages[imageIndex]), cv2.IMREAD_UNCHANGED)
        if tshirtImg is not None:
            # Ensure T-shirt image has 4 channels
            if tshirtImg.shape[2] == 3:  # Convert to BGRA if needed
                tshirtImg = cv2.cvtColor(tshirtImg, cv2.COLOR_BGR2BGRA)
 
            # Get keypoints for T-shirt alignment
            lm11 = lmlist[12]  # Left Shoulder
            lm12 = lmlist[11]  # Right Shoulder
            lm23 = lmlist[23]  # Left Hip
            lm24 = lmlist[24]  # Right Hip
 
            # Calculate T-shirt size and position
            shoulder_width = abs(lm12[0] - lm11[0])
            shirt_width = int(shoulder_width * fixedRatio * scale_factor)
            shirt_height = int((abs(lm23[1] - lm11[1]) + abs(lm24[1] - lm12[1])) / 2 * scale_factor)
 
            if shirt_width > 0 and shirt_height > 0:
                resizedTshirt = cv2.resize(tshirtImg, (shirt_width, shirt_height))
 
                # Top-left position for overlay
                top_left_x = int(lm11[0] - (shirt_width - shoulder_width) / 2)
                top_left_y = int(lm11[1]) - int(shirt_height * 0.3) + offset_y
 
                # Constrain overlay within the image frame
                img_height, img_width, _ = img.shape
                if top_left_x + shirt_width > img_width:
                    shirt_width = img_width - top_left_x
                if top_left_y + shirt_height > img_height:
                    shirt_height = img_height - top_left_y
 
                # Overlay the T-shirt on the frame
                img = cvzone.overlayPNG(img, resizedTshirt, [top_left_x, top_left_y])
        else:
            print("Error: Failed to load T-shirt image.")
   
    # Display the output
    cv2.imshow("Virtual Dressing Room", img)
 
    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release resources
cap.release()
cv2.destroyAllWindows()
