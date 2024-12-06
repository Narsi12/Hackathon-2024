from flask import Flask, request, render_template, send_file,Response,jsonify
import mediapipe as mp
import numpy as np
import io
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import math
from PIL import Image




app = Flask(__name__)

# Foot measurement

@app.route('/foot')
def home():
    """Serve the HTML page for foot measurement."""
    return render_template('foot_measurement.html')


@app.route('/calculate-foot-length', methods=['POST'])
def calculate_foot_length():
    """
    API endpoint to calculate foot length from an uploaded image.
    """
    # Check if the file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Read the image
    image_data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # Check image size and resize if necessary
    target_width, target_height = 522, 406
    if image.shape[1] != target_width or image.shape[0] != target_height:
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return jsonify({"error": "No contours found. Ensure the foot is clearly visible in the image."}), 400

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit a bounding rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Calculate the longest dimension of the rectangle as the foot length in pixels
    foot_length_pixels = max(
        math.dist(box[0], box[1]),
        math.dist(box[1], box[2]),
        math.dist(box[2], box[3]),
        math.dist(box[3], box[0])
    )

    # Reference length for scaling
    reference_length_cm = 2.25  # Reference object size in centimeters
    reference_length_pixels = 50  # Reference object size in pixels (manually measured)
    scaling_factor = reference_length_cm / reference_length_pixels

    # Convert to centimeters and inches
    foot_length_cm = foot_length_pixels * scaling_factor
    foot_length_inches = foot_length_cm / 2.54

    # Annotate the image
    annotated_image = image.copy()
    cv2.drawContours(annotated_image, [box], -1, (0, 255, 0), 2)
    cv2.putText(
        annotated_image,
        f"Length: {foot_length_inches:.2f} inches",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Convert the annotated image to a format suitable for sending in response
    _, buffer = cv2.imencode('.png', annotated_image)
    image_bytes = io.BytesIO(buffer)

    # Return the result
    return Response(
        image_bytes.getvalue(),
        mimetype='image/png',
        headers={
            "X-Foot-Length-Inches": f"{foot_length_inches:.2f}"
        }
    )
    
    
    
# AI-Video Dress Trail

# Global variable to manage webcam state
webcam_active = False
cap = None  # VideoCapture object

@app.route('/')
def index_ai_video():
    """Home page where users select the video source and upload the T-shirt."""
    return render_template('video.html')

def process_video(tshirt_image):
    """Process the live webcam feed with pose detection and overlay T-shirt."""
    global webcam_active, cap
    detector = PoseDetector(detectionCon=0.5, trackCon=0.5)
    fixed_ratio = 260 / 190
    scale_factor = 1.5
    offset_y = 30

    while webcam_active and cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=True)
        lmlist, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

        if len(lmlist) >= 25:  # Ensure all keypoints are detected
            lm11 = lmlist[12]  # Left Shoulder
            lm12 = lmlist[11]  # Right Shoulder
            lm23 = lmlist[23]  # Left Hip
            lm24 = lmlist[24]  # Right Hip

            shoulder_width = abs(lm12[0] - lm11[0])
            shirt_width = int(shoulder_width * fixed_ratio * scale_factor)
            shirt_height = int((abs(lm23[1] - lm11[1]) + abs(lm24[1] - lm12[1])) / 2 * scale_factor)

            if shirt_width > 0 and shirt_height > 0:
                resized_tshirt = cv2.resize(tshirt_image, (shirt_width, shirt_height))
                top_left_x = int(lm11[0] - (shirt_width - shoulder_width) / 2)
                top_left_y = int(lm11[1]) - int(shirt_height * 0.3) + offset_y
                img = cvzone.overlayPNG(img, resized_tshirt, [top_left_x, top_left_y])

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start the live webcam feed or video stream."""
    global webcam_active, cap
    tshirt_file = request.files['tshirt']
    video_file = request.files.get('video')  # Get the uploaded video file, if any

    if not tshirt_file:
        return "T-shirt image is required!", 400

    tshirt_image = cv2.imdecode(np.frombuffer(tshirt_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if tshirt_image is None:
        return "Invalid T-shirt image!", 400

    # If video is uploaded, use it as the source
    if video_file:
        video_path = "/path/to/save/video.mp4"  # Save the video file temporarily
        video_file.save(video_path)
        cap = cv2.VideoCapture(video_path)
    else:
        # Default to the webcam feed
        if not webcam_active:
            webcam_active = True
            cap = cv2.VideoCapture(0)  # Start capturing from webcam
            
            # Set higher resolution for the webcam feed
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

    return Response(process_video(tshirt_image),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop the webcam feed."""
    global webcam_active, cap
    webcam_active = False
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({"status": "Webcam stopped"})




# 2d shoetry on 

height_p = 231
width_p = 367
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def detect_foot_coordinates(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_foot = {
            "ankle": (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y),
        }
        right_foot = {
            "ankle": (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y),
        }
        return left_foot, right_foot, image
    return None, None, image

def overlay_image(background, overlay, position):
    y, x = position
    h, w, _ = overlay.shape

    # Adjust position if overlay exceeds image bounds
    if y + h > background.shape[0]:
        y = background.shape[0] - h
    if x + w > background.shape[1]:
        x = background.shape[1] - w

    # Ensure the position is valid
    y = max(0, y)
    x = max(0, x)

    # Extract the regions
    bg_region = background[y:y+h, x:x+w]
    alpha = overlay[:, :, 3] / 255.0  # Extract alpha channel
    for c in range(0, 3):
        bg_region[:, :, c] = (1 - alpha) * bg_region[:, :, c] + alpha * overlay[:, :, c]

    background[y:y+h, x:x+w] = bg_region
    return background

@app.route('/shoes')
def index_2d_shoes():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    person_image = request.files['person_image']
    left_shoe_image = request.files['left_shoe_image']
    right_shoe_image = request.files['right_shoe_image']

    # Convert uploaded images to numpy arrays
    person_image_bytes = np.frombuffer(person_image.read(), np.uint8)
    left_shoe_bytes = np.frombuffer(left_shoe_image.read(), np.uint8)
    right_shoe_bytes = np.frombuffer(right_shoe_image.read(), np.uint8)

    # Decode images from bytes
    person_image = cv2.imdecode(person_image_bytes, cv2.IMREAD_COLOR)
    left_shoe = cv2.imdecode(left_shoe_bytes, cv2.IMREAD_UNCHANGED)
    right_shoe = cv2.imdecode(right_shoe_bytes, cv2.IMREAD_UNCHANGED)

    # Resize the person image
    resized_person_image = resize_image(person_image, height_p, width_p)

    # Detect foot coordinates
    left_foot, right_foot, person_image = detect_foot_coordinates(resized_person_image)
    if not left_foot or not right_foot:
        return "Failed to detect pose landmarks.", 400

    # Resize shoes
    height, width, _ = person_image.shape
    shoe_width = int(width * 0.19)
    shoe_height = int(height * 0.16)
    left_shoe = resize_image(left_shoe, shoe_width, shoe_height)
    right_shoe = resize_image(right_shoe, shoe_width, shoe_height)

    # Get foot coordinates in pixels
    left_ankle_px = (int(left_foot['ankle'][0] * width), int(left_foot['ankle'][1] * height))
    right_ankle_px = (int(right_foot['ankle'][0] * width), int(right_foot['ankle'][1] * height))

    # Overlay the shoes on the person image
    output_image = overlay_image(person_image, left_shoe, (left_ankle_px[1] - shoe_height // 2, left_ankle_px[0] - shoe_width // 2))
    output_image = overlay_image(output_image, right_shoe, (right_ankle_px[1] - shoe_height // 2, right_ankle_px[0] - shoe_width // 2))

    # Convert final image to bytes and return as a response
    _, img_encoded = cv2.imencode('.jpg', output_image)
    img_bytes = img_encoded.tobytes()

    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

