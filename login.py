from flask import Flask, request, Response, render_template_string
from cryptography.fernet import Fernet
import os, cv2, time
from datetime import datetime

app = Flask(__name__)

USERNAME = "admin"
PASSWORD = "12345"
IMAGE_DIR = "detected_footage"
KEY_FILE = "private_key.key"

# Generate Private Key (if missing)
if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    open(KEY_FILE,"wb").write(key)
    print("🔑 New Private Key Created!")
else:
    print("🔐 Private Key Loaded")

# ------------------------- ENCRYPT IMAGE (USED BY DETECTOR) ------------------------- #
def encrypt_image(frame, weapon,label_confidence):
    key = open(KEY_FILE,"rb").read()
    cipher = Fernet(key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{IMAGE_DIR}/{weapon}_{timestamp}.enc"
    
    success,buffer=cv2.imencode(".jpg",frame)
    if success:
        encrypted = cipher.encrypt(buffer.tobytes())
        open(filename,"wb").write(encrypted)
        print("🔐 Saved Encrypted →",filename)
    return filename

# ------------------------- LOGIN PAGE ------------------------- #
@app.route("/")
def login_page():
    return """
    <h2>🔐 Secure Access</h2>
    <form method="POST" action="/login">
        Username: <input name="u"><br><br>
        Password: <input name="p" type="password"><br><br>
        <button>Login</button>
    </form>
    """

# ------------------------- VERIFY LOGIN ------------------------- #
@app.route("/login", methods=["POST"])
def login():
    if request.form["u"] == USERNAME and request.form["p"] == PASSWORD:
        return """
        <h2>Enter Private Key 🔑</h2>
        <form action="/unlock" method="POST">
            <input type="password" name="key" style="width:300px">
            <button>Unlock Images</button>
        </form>
        """
    return "<h3>❌ Wrong Credentials</h3>"

# ------------------------- UNLOCK & SHOW IMAGES ------------------------- #
@app.route("/unlock", methods=["POST"])
def unlock():
    entered_key = request.form["key"].encode()
    
    try:
        Fernet(entered_key) # Validate user provided key
    except:
        return "<h3>❌ Incorrect Private Key</h3>"

    files=[f for f in os.listdir(IMAGE_DIR) if f.endswith(".enc")]
    if not files:
        return "<h2>No Encrypted Images Found</h2>"

    page="<h2>📂 Decrypted Image Viewer</h2>"
    for f in files:
        page+=f"<img src='/img/{f}' width='300px' style='margin:10px;'>"
    return page

# ------------------------- SERVE DECRYPTED IMAGE ------------------------- #
@app.route("/img/<filename>")
def serve(filename):
    key=open(KEY_FILE,"rb").read()
    cipher=Fernet(key)

    enc = open(f"{IMAGE_DIR}/{filename}","rb").read()
    decrypted = cipher.decrypt(enc)

    return Response(decrypted,mimetype="image/jpeg")

# ------------------------------------------------------------------------ #
if __name__=="__main__":
    app.run(port=8000,debug=True)
