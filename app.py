from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
import hashlib
import csv
import time
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
from deepface import DeepFace
import mediapipe as mp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants
USER_DB = "users.csv"
VC_EXPIRY_TIME = 24 * 60 * 60
MAX_FAILED_ATTEMPTS = 3
OTP_DB = {}

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "khare.ritik01@gmail.com"
EMAIL_PASSWORD = "bpqz vqhy xydm utgb"

# Load FaceNet model
model_path = 'facenet_model.pb'

try:
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    sess = tf.compat.v1.Session(graph=graph)
    input_tensor = graph.get_tensor_by_name("input:0")
    output_tensor = graph.get_tensor_by_name("embeddings:0")
    phase_train_tensor = graph.get_tensor_by_name("phase_train:0")
except Exception as e:
    logger.error(f"Failed to load FaceNet model: {e}")
    raise

# Initialize face detection models
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
try:
    face_predictor = dlib.shape_predictor(predictor_path)
except Exception as e:
    logger.error(f"Failed to load shape predictor: {e}")
    raise
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# with tf.io.gfile.GFile(model_path, "rb") as f:
#     graph_def = tf.compat.v1.GraphDef()
#     graph_def.ParseFromString(f.read())

# with tf.Graph().as_default() as graph:
#     tf.import_graph_def(graph_def, name="")
# sess = tf.compat.v1.Session(graph=graph)
# input_tensor = graph.get_tensor_by_name("input:0")
# output_tensor = graph.get_tensor_by_name("embeddings:0")
# phase_train_tensor = graph.get_tensor_by_name("phase_train:0")

# # Initialize face detection models
# detector = dlib.get_frontal_face_detector()
# predictor_path = "shape_predictor_68_face_landmarks.dat"
# face_predictor = dlib.shape_predictor(predictor_path)
# haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_cascade_path)
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Camera management
cap = None

def initialize_camera():
    global cap
    if cap is None or not cap.isOpened():
        logger.info("Initializing camera...")
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                break
            time.sleep(0.1)
    if not cap.isOpened():
        logger.error("Camera failed to initialize")
    return cap.isOpened()

# def initialize_camera():
#     global cap
#     if cap is None or not cap.isOpened():
#         logger.info("Initializing camera...")
#         cap = cv2.VideoCapture(0)
#         cap.set(3, 640)
#         cap.set(4, 480)
#         for _ in range(10):
#             ret, _ = cap.read()
#             if ret:
#                 break
#             time.sleep(0.1)
#     return cap.isOpened()

def release_camera():
    global cap
    if cap is not None and cap.isOpened():
        logger.info("Releasing camera...")
        cap.release()
        cap = None

# Utility Functions (unchanged from your Flask code)
def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())
    except Exception as e:
        logger.error(f"Email sending failed: {e}")

def generate_first_share(user_id, password, img_size=(100, 100)):
    seed = int(hashlib.sha256((user_id + password).encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, img_size, dtype=np.uint8) * 255)

def generate_second_share(first_share):
    rng = np.random.default_rng()
    second_share = (rng.integers(0, 2, first_share.shape, dtype=np.uint8) * 255)
    combined_share = cv2.bitwise_xor(first_share, second_share)
    return second_share, hashlib.sha256(combined_share.tobytes()).hexdigest()

def user_exists(user_id):
    if not os.path.exists(USER_DB):
        return False
    with open(USER_DB, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == user_id:
                return True
    return False

# def register_user(user_id, password, email):
#     if user_exists(user_id):
#         return None
#     first_share = generate_first_share(user_id, password)
#     second_share, hashed_combined = generate_second_share(first_share)
#     with open(USER_DB, "a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([user_id, hashlib.sha256(password.encode()).hexdigest(), f"{user_id}_vc_share.png", hashed_combined, email, time.time(), 0])
#     send_email(email, "Welcome! Registration Successful", "Your account has been successfully registered.")
#     cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_vc_share.png"), second_share)
#     return second_share

def register_user(user_id, password, email):
    try:
        if user_exists(user_id):
            logger.warning(f"User {user_id} already exists")
            return None
        first_share = generate_first_share(user_id, password)
        second_share, hashed_combined = generate_second_share(first_share)
        with open(USER_DB, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([user_id, hashlib.sha256(password.encode()).hexdigest(), f"{user_id}_vc_share.png", hashed_combined, email, time.time(), 0])
        send_email(email, "Welcome! Registration Successful", "Your account has been successfully registered.")
        vc_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_vc_share.png")
        logger.info(f"Saving VC share to {vc_path}")
        cv2.imwrite(vc_path, second_share)
        if not os.path.exists(vc_path):
            logger.error(f"Failed to save VC share to {vc_path}")
            raise Exception("VC share save failed")
        return second_share
    except Exception as e:
        logger.error(f"Error in register_user: {e}")
        raise

# def capture_face_embeddings(student_name):
#     student_folder = os.path.join("students", student_name)
#     os.makedirs(student_folder, exist_ok=True)
#     if not initialize_camera():
#         logger.error("Failed to initialize camera for face embedding capture")
#         return 0
#     max_images = 50
#     count = 0
#     embeddings_list = []
    
#     while count < max_images:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning("Failed to capture frame")
#             continue
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
#         for (x, y, w, h) in faces:
#             if w > 100 and h > 100:
#                 face_img = frame[y:y + h, x:x + w]
#                 if is_face_visible(face_img):
#                     face_img = preprocess_face(face_img)
#                     face_img = cv2.resize(face_img, (160, 160))
#                     face_img = (face_img - 127.5) / 128.0
#                     embedding = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(face_img, axis=0), phase_train_tensor: False})
#                     embeddings_list.append((embedding, face_img))
#                     count += 1
#     embeddings_list.sort(key=lambda x: np.linalg.norm(x[0]))
#     best_embeddings = embeddings_list[:10]
#     for i, (embedding, img) in enumerate(best_embeddings):
#         np.save(os.path.join(student_folder, f"{student_name}_{i}_embeddings.npy"), embedding)
#         cv2.imwrite(os.path.join(student_folder, f"{student_name}_{i}.jpg"), img * 128.0 + 127.5)
#     release_camera()
#     return len(best_embeddings)

def capture_face_embeddings(student_name):
    try:
        student_folder = os.path.join("students", student_name)
        os.makedirs(student_folder, exist_ok=True)
        # Show camera preview and wait for Enter key
        logger.info("Showing camera preview. Press Enter to start capturing embeddings.")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Align your face and press Enter to start", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                logger.info("Enter key pressed. Starting embedding capture.")
                break
            elif key == 27:  # Escape key to cancel
                logger.info("Capture cancelled by user.")
                release_camera()
                return 0

        max_images = 50
        count = 0
        embeddings_list = []
        
        while count < max_images and session.get('register_state') == 'capturing':
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                if w > 100 and h > 100:
                    face_img = frame[y:y + h, x:x + w]
                    if is_face_visible(face_img):
                        face_img = preprocess_face(face_img)
                        face_img = cv2.resize(face_img, (160, 160))
                        face_img = (face_img - 127.5) / 128.0
                        embedding = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(face_img, axis=0), phase_train_tensor: False})
                        embeddings_list.append((embedding, face_img))
                        count += 1
        embeddings_list.sort(key=lambda x: np.linalg.norm(x[0]))
        best_embeddings = embeddings_list[:10]
        for i, (embedding, img) in enumerate(best_embeddings):
            np.save(os.path.join(student_folder, f"{student_name}_{i}_embeddings.npy"), embedding)
            cv2.imwrite(os.path.join(student_folder, f"{student_name}_{i}.jpg"), img * 128.0 + 127.5)
        release_camera()
        return len(best_embeddings)
    except Exception as e:
        logger.error(f"Error in capture_face_embeddings: {e}")
        release_camera()
        raise   

def send_otp(email):
    otp = random.randint(100000, 999999)
    OTP_DB[email] = otp
    send_email(email, "VC Share Regeneration OTP", f"Your OTP for VC share regeneration is: {otp}")
    return otp

def regenerate_vc(user_id, password, email, otp_input):
    if OTP_DB.get(email) != int(otp_input):
        return False
    first_share = generate_first_share(user_id, password)
    second_share, hashed_combined = generate_second_share(first_share)
    with open(USER_DB, "r") as file:
        users = list(csv.reader(file))
    for row in users:
        if row and row[0] == user_id:
            row[2] = f"{user_id}_vc_share.png"
            row[3] = hashed_combined
            row[5] = str(time.time())
            row[6] = "0"
    with open(USER_DB, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(users)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_vc_share.png"), second_share)
    return True

def verify_vc_shares(user_id, password, user_vc_share, stored_hashed_combined, email, user_row):
    logger.info(f"Verifying VC for user {user_id}")
    first_share = generate_first_share(user_id, password)
    file_bytes = np.asarray(bytearray(user_vc_share.read()), dtype=np.uint8)
    user_share = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if user_share is None or user_share.shape != first_share.shape:
        logger.warning(f"Invalid VC share for user {user_id}")
        return False
    reconstructed = cv2.bitwise_xor(first_share, user_share)
    hashed_reconstructed = hashlib.sha256(reconstructed.tobytes()).hexdigest()
    if hashed_reconstructed == stored_hashed_combined:
        logger.info(f"VC verification successful for user {user_id}")
        return True
    else:
        logger.warning(f"VC verification failed for user {user_id}")
        user_row[6] = str(int(user_row[6]) + 1)
        with open(USER_DB, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows([row if row[0] != user_id else user_row for row in csv.reader(open(USER_DB, "r"))])
        send_email(email, "Alert: Incorrect VC Attempt", "Someone tried accessing your account with an incorrect VC share.")
        return False

def verify_user(user_id, password):
    if not user_exists(user_id):
        return None
    with open(USER_DB, "r") as file:
        users = list(csv.reader(file))
    for row in users:
        if row and row[0] == user_id:
            stored_password, stored_vc_path, stored_hashed_combined, email, last_vc_time, failed_attempts = row[1:]
            last_vc_time, failed_attempts = float(last_vc_time), int(failed_attempts)
            if hashlib.sha256(password.encode()).hexdigest() != stored_password:
                send_email(email, "Alert: Incorrect Password Attempt", "Someone tried to access your account with the wrong password.")
                return None
            if time.time() - last_vc_time > VC_EXPIRY_TIME or failed_attempts >= MAX_FAILED_ATTEMPTS:
                return {"expired": True, "email": email, "row": row}
            return {"stored_hashed_combined": stored_hashed_combined, "email": email, "row": row}
    return None

def is_face_visible(face_img):
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_face) > 50

def preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(enhanced, table)

def process_face(face_img):
    face_img = preprocess_face(face_img)
    face_img = cv2.resize(face_img, (160, 160))
    face_img = (face_img - 127.5) / 128.0
    return sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(face_img, axis=0), phase_train_tensor: False})

def match_face(embedding, user_embeddings, threshold=0.9):
    if not user_embeddings:
        return False
    min_dist = float("inf")
    for stored_embedding in user_embeddings:
        dist = np.linalg.norm(embedding - stored_embedding)
        min_dist = min(min_dist, dist)
    return min_dist < threshold

def load_user_embeddings(username):
    user_path = os.path.join("students", username)
    if not os.path.exists(user_path):
        return []
    return [np.load(os.path.join(user_path, f)) for f in os.listdir(user_path) if f.endswith("_embeddings.npy")]

def display_text(image, text, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    position = (30, 50)
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return redirect(url_for('verify'))
    if 'state' not in session or session['state'] == 'waiting':
        return redirect(url_for('face_verify'))
    initialize_camera()
    user_id = session['user_id']
    initial_state = session['state']
    email = session.get('email', 'unknown@example.com')
    return Response(gen_frames(user_id, initial_state, email), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(user_id, initial_state, email):
    if not initialize_camera():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', display_text(np.zeros((480, 640, 3), dtype=np.uint8), "Camera Error", (0, 0, 255)))[1].tobytes() + b'\r\n')
        return

    user_embeddings = load_user_embeddings(user_id)
    state = initial_state
    face_detected = False
    face_recognized = False
    liveness_confirmed = False
    emotion_verified = False
    random_instruction = random.choice(["Blink your eyes", "Open your mouth"])
    start_time = time.time()
    verification_failed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = display_text(frame, "No Camera Feed", (0, 0, 255))

        if state == "waiting":
            frame = display_text(frame, "Press 'Start Verification' to Begin", (0, 255, 255))

        elif state == "pending":
            frame = display_text(frame, "Press Enter to Start Verification", (0, 255, 255))

        elif state == "face_detection":
            faces = detector(frame)
            if faces:
                face_detected = True
                state = "face_recognition"
                frame = display_text(frame, "Face Detected! Verifying Identity...", (0, 255, 0))
            else:
                frame = display_text(frame, "Align your face with the camera", (0, 140, 255))

        elif state == "face_recognition":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if faces:
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_img = frame[y:y + h, x:x + w]
                    if face_img.size > 0:
                        try:
                            embedding = process_face(face_img)
                            face_recognized = match_face(embedding, user_embeddings)
                            if face_recognized:
                                state = "liveness"
                                frame = display_text(frame, "Face Verified! Perform Liveness Check...", (0, 255, 0))
                            else:
                                frame = display_text(frame, "Face Not Recognized", (0, 0, 255))
                        except Exception as e:
                            logger.error(f"Error processing face: {e}")
            if not face_recognized and time.time() - start_time > 5:
                verification_failed = True
                send_email(email, "Alert: Face Verification Failed", f"Face verification failed for user {user_id}.")
                break

        elif state == "liveness":
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    left_eye_idx = [159, 145]
                    right_eye_idx = [386, 374]
                    mouth_idx = [13, 14]
                    left_eye_dist = abs(landmarks[left_eye_idx[0]].y - landmarks[left_eye_idx[1]].y)
                    right_eye_dist = abs(landmarks[right_eye_idx[0]].y - landmarks[right_eye_idx[1]].y)
                    mouth_distance = abs(landmarks[mouth_idx[0]].y - landmarks[mouth_idx[1]].y)
                    if (random_instruction == "Blink your eyes" and left_eye_dist < 0.01 and right_eye_dist < 0.01) or \
                       (random_instruction == "Open your mouth" and mouth_distance > 0.05):
                        liveness_confirmed = True
                        state = "emotion"
                        frame = display_text(frame, "Liveness Confirmed! Checking Emotion...", (0, 255, 0))
            cv2.putText(frame, f"Perform Action: {random_instruction}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
            if not liveness_confirmed and time.time() - start_time > 10:
                verification_failed = True
                send_email(email, "Alert: Liveness Check Failed", f"Liveness check failed for user {user_id}.")
                break

        elif state == "emotion":
            try:
                emotion_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = emotion_analysis[0]['dominant_emotion']
                if emotion in ["happy", "neutral"]:
                    emotion_verified = True
                    frame = display_text(frame, "Access Granted!", (0, 255, 0))
                    state = 'complete'
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    break
                else:
                    frame = display_text(frame, "Emotion Mismatch!", (0, 0, 255))
                    verification_failed = True
                    send_email(email, "Alert: Emotion Mismatch", f"Emotion mismatch detected for user {user_id}.")
                    break
            except Exception as e:
                logger.error(f"Error in emotion detection: {e}")
                frame = display_text(frame, "Emotion Detection Failed", (0, 0, 255))
                verification_failed = True
                send_email(email, "Alert: Emotion Detection Failed", f"Emotion detection failed for user {user_id}.")
                break

        if time.time() - start_time > 15:
            verification_failed = True
            send_email(email, "Alert: Verification Timeout", f"Verification timed out for user {user_id}.")
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    release_camera()
    if verification_failed:
        yield (b'--frame\r\n'
               b'Content-Type: text/html\r\n\r\n'
               b'<html><body><script>window.location.href="{{ url_for(\'face_verify\') }}";</script></body></html>\r\n')
    elif state == 'complete':
        yield (b'--frame\r\n'
               b'Content-Type: text/html\r\n\r\n'
               b'<html><body><script>window.location.href="{{ url_for(\'access_granted\') }}";</script></body></html>\r\n')

# Routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register_video_feed')
def register_video_feed():
    if 'user_id' not in session:
        return redirect(url_for('register'))
    if not initialize_camera():
        return Response("Camera failed to initialize", mimetype='text/plain')
    return Response(gen_register_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_register_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            frame = display_text(np.zeros((480, 640, 3), dtype=np.uint8), "No Camera Feed", (0, 0, 255))
        else:
            state = session.get('register_state', 'waiting')
            if state == 'waiting':
                frame = display_text(frame, "Press 'Start Capture' or Enter to Begin", (0, 255, 255))
            elif state == 'capturing':
                frame = display_text(frame, f"Capturing Embeddings... ({session.get('embedding_count', 0)}/50)", (0, 255, 0))
            elif state == 'complete':
                frame = display_text(frame, "Registration Complete!", (0, 255, 0))
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                break

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         user_id = request.form.get('user_id')
#         password = request.form.get('password')
#         email = request.form.get('email')
#         if not all([user_id, password, email]):
#             flash('Please fill in all fields.', 'danger')
#             return redirect(url_for('register'))
#         if register_user(user_id, password, email):
#             embeddings_count = capture_face_embeddings(user_id)
#             flash(f"Registration successful! {embeddings_count} face embeddings captured.", 'success')
#             return redirect(url_for('home'))
#         else:
#             flash('User already exists.', 'danger')
#     return render_template('register.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'action' not in request.form:  # Initial form submission
            user_id = request.form.get('user_id')
            password = request.form.get('password')
            email = request.form.get('email')
            logger.info(f"Register attempt: user_id={user_id}, email={email}")
            if not all([user_id, password, email]):
                flash('Please fill in all fields.', 'danger')
                logger.warning("Missing form fields")
                return redirect(url_for('register'))
            try:
                second_share = register_user(user_id, password, email)
                if second_share is not None:
                    session['user_id'] = user_id
                    session['register_state'] = 'waiting'
                    return redirect(url_for('register'))  # Redirect to show camera
                else:
                    flash('User already exists.', 'danger')
                    logger.warning(f"User {user_id} already exists")
            except Exception as e:
                flash(f"Registration failed: {str(e)}", 'danger')
                logger.error(f"Registration failed for {user_id}: {e}")
        elif request.form['action'] == 'start_capture':  # Start capturing embeddings
            if 'user_id' in session:
                session['register_state'] = 'capturing'
                session['embedding_count'] = 0
                embeddings_count = capture_face_embeddings(session['user_id'])
                session['register_state'] = 'complete'
                session['embedding_count'] = embeddings_count
                flash(f"Registration successful! {embeddings_count} face embeddings captured.", 'success')
                logger.info(f"User {session['user_id']} registered successfully with {embeddings_count} embeddings")
                session.pop('register_state', None)
                session.pop('embedding_count', None)
                session.pop('user_id', None)
                return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')
        result = verify_user(user_id, password)
        if not result:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('verify'))
        if 'expired' in result:
            session['user_id'] = user_id
            session['password'] = password
            session['email'] = result['email']
            send_otp(result['email'])
            return redirect(url_for('otp_verify'))
        session['user_id'] = user_id
        session['password'] = password
        session['stored_hashed_combined'] = result['stored_hashed_combined']
        session['email'] = result['email']
        session['user_row'] = result['row']
        session['vc_verified'] = False  # From Streamlit
        return redirect(url_for('vc_verify'))
    return render_template('verify.html')

@app.route('/otp_verify', methods=['GET', 'POST'])
def otp_verify():
    if request.method == 'POST':
        otp_input = request.form.get('otp')
        if regenerate_vc(session['user_id'], session['password'], session['email'], otp_input):
            flash('New VC share generated. Please save it.', 'success')
            return redirect(url_for('verify'))
        flash('Incorrect OTP.', 'danger')
    return render_template('otp_verify.html')

@app.route('/vc_verify', methods=['GET', 'POST'])
def vc_verify():
    if 'user_id' not in session:
        return redirect(url_for('verify'))
    if request.method == 'POST':
        if 'vc_share' not in request.files:
            flash('No VC share uploaded.', 'danger')
            return redirect(url_for('vc_verify'))
        vc_share = request.files['vc_share']
        if vc_share.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('vc_verify'))
        if vc_share and verify_vc_shares(session['user_id'], session['password'], vc_share, session['stored_hashed_combined'], session['email'], session['user_row']):
            session['vc_verified'] = True
            session['state'] = 'waiting'
            return redirect(url_for('face_verify'))
        else:
            flash('VC verification failed.', 'danger')
            return redirect(url_for('vc_verify'))
    return render_template('vc_verify.html')

@app.route('/face_verify', methods=['GET', 'POST'])
def face_verify():
    if 'user_id' not in session:
        return redirect(url_for('verify'))
    if not session.get('vc_verified', False):  # Ensure VC is verified
        return redirect(url_for('vc_verify'))
    if request.method == 'POST' and request.form.get('action') == 'start':
        session['state'] = 'pending'
        return redirect(url_for('face_verify'))
    if 'state' not in session:
        session['state'] = 'waiting'
    return render_template('face_verify.html')

@app.route('/start_verification', methods=['POST'])
def start_verification():
    if 'user_id' not in session or session['user_id'] != request.json.get('user_id'):
        return redirect(url_for('verify'))
    session['state'] = 'face_detection'
    return redirect(url_for('face_verify'))

@app.route('/access_granted')
def access_granted():
    if 'user_id' not in session or session.get('state') != 'complete':
        return redirect(url_for('verify'))
    session.pop('state', None)
    session.pop('vc_verified', None)
    return render_template('access_granted.html')

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000, use_reloader=False)
    finally:
        release_camera()
        cv2.destroyAllWindows()