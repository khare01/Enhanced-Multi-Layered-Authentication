# Enhanced-Multi-Layered-Authentication




A robust and modern authentication system combining **Visual Cryptography**, **Password Verification**, **Face Recognition**, **Liveness Detection**, **Emotion Analysis**, and **OTP-based VC Regeneration** to ensure advanced security and identity validation.

## 🚀 Features

* **Password-based Login**
* **Visual Cryptography (VC) Verification**

  * User uploads VC Share 1
  * Share 2 is generated using password
  * Both are stacked and matched securely
* **Face Recognition** using FaceNet embeddings
* **Liveness Detection** (Blink or Mouth Open using MediaPipe)
* **Emotion Detection** (via DeepFace) for enhanced verification
* **Email Alert System** for failed or suspicious attempts
* **OTP-based VC Regeneration** if VC fails or expires
* **Session Management** and timeout handling
* **Lighting Adjustment** using CLAHE and gamma correction

## 🧠 Tech Stack

* **Backend:** Flask (Python)
* **Face Recognition:** FaceNet
* **Liveness Detection:** MediaPipe (FaceMesh)
* **Emotion Analysis:** DeepFace
* **Visual Cryptography:** Custom image XOR + PIL
* **Frontend:** HTML, CSS (Bootstrap), JavaScript
* **Image Processing:** OpenCV, NumPy, PIL
* **Email Alerts:** SMTP
* **Database:** SQLite (for user data & logging)

## 📷 Authentication Flow

1. **Login** with Username & Password
2. **VC Verification**

   * User uploads Share 1
   * System generates Share 2 using password
   * Stacked image compared with stored hash
3. **Camera Access** (Live Feed)
4. **Face Detection and Verification**
5. **Liveness Check** (Blink/Mouth Open)
6. **Emotion Check** (e.g., "Happy" or "Neutral")
7. **Final Access Granted** or failure triggers:

   * OTP for VC regeneration
   * Email alert

## 📁 Folder Structure

```bash
Enhanced-Multi-Layered-Auth/
├── app.py                  # Main Flask application
├── static/                 # CSS, JS, and images
├── templates/              # HTML files
└── README.md
```

## ⚙️ Installation & Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/your-username/enhanced-auth.git
cd enhanced-auth
```

2. **Create and activate virtual environment**

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.



## 🔐 Security Highlights

* **Time-based VC share validity**
* **Email alerts on suspicious activity**
* **Anti-spoofing via liveness check**
* **Multiple failed attempts trigger OTP regeneration**

## 🛠️ Future Enhancements

* Add voice recognition as an extra layer
* Integrate with external face verification APIs
* Use MongoDB or PostgreSQL for scalability
* Add role-based access for different user types

## 👨‍💻 Author

**Ritik Khare**
MCA Graduate, VIT Chennai


---

Let me know if you want this in a Markdown file or want to include a **video demo**, **live deployment link**, or **custom badges** (e.g., for Python version, license, etc.).
