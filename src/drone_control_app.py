import cv2
import numpy as np
from tensorflow import keras
import serial
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
from collections import deque

# Configuration
MODEL_PATH = '../pilot_presence_model.h5'
SERIAL_PORT = '/dev/cu.usbmodem00000000001B1' # serial port of drone controller
BAUD_RATE = 400000
IMG_SIZE = 128

# CRSF Protocol
CRSF_SYNC = 0xC8
CRSF_FRAMETYPE_RC_CHANNELS = 0x16
CRSF_CHANNEL_VALUE_MIN = 172
CRSF_CHANNEL_VALUE_MID = 992
CRSF_CHANNEL_VALUE_MAX = 1811


class CRSFPacket:
    #Creates packets to send commands to RadioMaster
    def __init__(self):
        self.channels = [CRSF_CHANNEL_VALUE_MID] * 16

    def set_channel(self, channel, value):
        if 0 <= channel < 16:
            self.channels[channel] = int(np.clip(value, CRSF_CHANNEL_VALUE_MIN, CRSF_CHANNEL_VALUE_MAX))

    def build_packet(self):
        packed = 0
        for i, ch in enumerate(self.channels):
            packed |= (ch & 0x7FF) << (i * 11)
        channel_bytes = [((packed >> (i * 8)) & 0xFF) for i in range(22)]
        payload = bytes([CRSF_FRAMETYPE_RC_CHANNELS] + channel_bytes)
        frame_length = len(payload) + 2
        crc = 0
        for byte in [frame_length] + list(payload):
            crc ^= byte
            for _ in range(8):
                crc = (crc << 1) ^ 0xD5 if crc & 0x80 else crc << 1
            crc &= 0xFF
        return bytes([CRSF_SYNC, frame_length]) + payload + bytes([crc])


class PilotPresenceDetector:
    def __init__(self, model_path):
        try:
            self.model = keras.models.load_model(model_path)
            self.model_loaded = True
            print(f"Pilot presence model loaded: {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            self.model_loaded = False

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prediction_history = deque(maxlen=10)
        self.last_face_time = time.time()

    def detect_pilot_presence(self, frame):
        pilot_present = False
        confidence = 0.0
        face_bbox = None

        if self.model_loaded:
            # Use CNN to detect pilot presence
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = self.model.predict(img, verbose=0)[0][0]

            # Class mapping: no_pilot=0, pilot_present=1
            # So if prediction > 0.5, pilot is present
            pilot_present = prediction > 0.5
            confidence = prediction if pilot_present else (1 - prediction)

            # Smooth predictions
            self.prediction_history.append(pilot_present)
            if len(self.prediction_history) >= 5:
                pilot_count = sum(self.prediction_history)
                pilot_present = pilot_count > len(self.prediction_history) / 2
        else:
            # Fallback to face detection if model not loaded
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            pilot_present = len(faces) > 0
            confidence = 1.0 if pilot_present else 0.0

        # Optional: Also use face detection for visual feedback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            face_areas = [(x, y, w, h, w * h) for (x, y, w, h) in faces]
            x, y, w, h, _ = max(face_areas, key=lambda f: f[4])
            face_bbox = (x, y, w, h)
            self.last_face_time = time.time()

        # Timeout if no activity for 3 seconds
        if time.time() - self.last_face_time > 3.0:
            pilot_present = False

        return pilot_present, confidence, face_bbox, len(faces) > 0 if 'faces' in locals() else False


class VirtualDrone:
    def __init__(self):
        self.altitude = 0.0
        self.x = 0.0
        self.y = 0.0
        self.battery = 100.0
        self.armed = False

    def update(self, throttle, pitch, roll, armed):
        self.armed = armed
        if armed:
            self.altitude += (throttle - 0.5) * 0.1
            self.altitude = max(0, min(10, self.altitude))
            self.x += (roll - 0.5) * 0.1
            self.y += (pitch - 0.5) * 0.1
            if self.altitude > 0:
                self.battery -= 0.01
                self.battery = max(0, self.battery)
        else:
            self.altitude = max(0, self.altitude - 0.2)


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, bg_color, fg_color, hover_color, **kwargs):
        width = kwargs.pop('width', 200)
        height = kwargs.pop('height', 50)
        super().__init__(parent, width=width, height=height, bg=parent['bg'],
                         highlightthickness=0, **kwargs)

        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.hover_color = hover_color
        self.text = text
        self.is_hovered = False

        self.draw_button()
        self.bind("<Button-1>", lambda e: self.command())
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def draw_button(self):
        self.delete("all")
        color = self.hover_color if self.is_hovered else self.bg_color

        # Rounded rectangle
        w, h = int(self['width']), int(self['height'])
        r = 15  # Radius

        self.create_arc(0, 0, 2 * r, 2 * r, start=90, extent=90, fill=color, outline="")
        self.create_arc(w - 2 * r, 0, w, 2 * r, start=0, extent=90, fill=color, outline="")
        self.create_arc(0, h - 2 * r, 2 * r, h, start=180, extent=90, fill=color, outline="")
        self.create_arc(w - 2 * r, h - 2 * r, w, h, start=270, extent=90, fill=color, outline="")
        self.create_rectangle(r, 0, w - r, h, fill=color, outline="")
        self.create_rectangle(0, r, w, h - r, fill=color, outline="")

        # Text
        self.create_text(w / 2, h / 2, text=self.text, fill=self.fg_color,
                         font=('Helvetica', 12, 'bold'))

    def on_enter(self, e):
        self.is_hovered = True
        self.draw_button()

    def on_leave(self, e):
        self.is_hovered = False
        self.draw_button()


class DroneControlApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Drone Authorization System")
        self.root.geometry("1100x650")
        self.root.configure(bg='#f5f5f5')

        # Components
        self.pilot_detector = PilotPresenceDetector(MODEL_PATH)
        self.drone = VirtualDrone()
        self.crsf_packet = CRSFPacket()

        # State
        self.is_authorized = False
        self.is_armed = False
        self.running = True
        self.serial_port = None

        # Camera
        self.cap = cv2.VideoCapture(0)

        # Setup UI
        self.setup_ui()

        # Start threads
        threading.Thread(target=self.update_video, daemon=True).start()
        threading.Thread(target=self.control_loop, daemon=True).start()

        # Connect serial
        self.connect_radiomaster()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#ffffff', height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        tk.Label(header, text="PilotWatch", bg='#ffffff',
                 fg='#2c2c2c', font=('Helvetica', 22, 'bold')).pack(side=tk.LEFT, padx=30, pady=20)

        self.status_indicator = tk.Canvas(header, width=20, height=20, bg='#ffffff', highlightthickness=0)
        self.status_indicator.pack(side=tk.RIGHT, padx=30)
        self.status_indicator.create_oval(2, 2, 18, 18, fill='#ff4444', outline='')

        # Alert banner (hidden by default)
        self.alert_banner = tk.Frame(self.root, bg='#ff4444', height=60)
        self.alert_label = tk.Label(self.alert_banner,
                                    text="⚠ PILOT LEFT STATION - FLIGHT SESSION ACTIVE ⚠",
                                    bg='#ff4444', fg='#ffffff',
                                    font=('Helvetica', 16, 'bold'))
        self.alert_label.pack(expand=True)

        # Main content area
        content = tk.Frame(self.root, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left - Video feed
        video_container = tk.Frame(content, bg='#ffffff', relief=tk.FLAT)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(video_container, bg='#000000')
        self.video_label.pack(padx=15, pady=15)

        self.auth_status = tk.Label(video_container, text="Waiting for pilot...",
                                    bg='#ffffff', fg='#666666', font=('Helvetica', 14))
        self.auth_status.pack(pady=(0, 15))

        # Right - Controls and info
        right_panel = tk.Frame(content, bg='#f5f5f5', width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # Pilot info card
        pilot_card = tk.Frame(right_panel, bg='#ffffff', relief=tk.FLAT)
        pilot_card.pack(fill=tk.X, pady=(0, 15))

        tk.Label(pilot_card, text="Pilot Status", bg='#ffffff', fg='#2c2c2c',
                 font=('Helvetica', 13, 'bold')).pack(anchor=tk.W, padx=20, pady=(15, 5))

        self.pilot_status_label = tk.Label(pilot_card, text="● Unauthorized",
                                           bg='#ffffff', fg='#ff4444',
                                           font=('Helvetica', 12))
        self.pilot_status_label.pack(anchor=tk.W, padx=20, pady=5)

        self.pilot_name = tk.Label(pilot_card, text="No pilot detected",
                                   bg='#ffffff', fg='#999999',
                                   font=('Helvetica', 11))
        self.pilot_name.pack(anchor=tk.W, padx=20, pady=(0, 15))

        # Drone status card
        drone_card = tk.Frame(right_panel, bg='#ffffff', relief=tk.FLAT)
        drone_card.pack(fill=tk.X, pady=(0, 15))

        tk.Label(drone_card, text="Flight Monitoring", bg='#ffffff', fg='#2c2c2c',
                 font=('Helvetica', 13, 'bold')).pack(anchor=tk.W, padx=20, pady=(15, 5))

        self.monitoring_status = tk.Label(drone_card, text="Monitoring Active",
                                          bg='#ffffff', fg='#00cc66',
                                          font=('Helvetica', 11))
        self.monitoring_status.pack(anchor=tk.W, padx=20, pady=2)

        self.connection_status = tk.Label(drone_card, text="Connection: Checking...",
                                          bg='#ffffff', fg='#666666',
                                          font=('Helvetica', 11))
        self.connection_status.pack(anchor=tk.W, padx=20, pady=2)

        self.session_label = tk.Label(drone_card, text="Session: Waiting",
                                      bg='#ffffff', fg='#666666',
                                      font=('Helvetica', 11))
        self.session_label.pack(anchor=tk.W, padx=20, pady=(2, 15))

        # Telemetry card
        telem_card = tk.Frame(right_panel, bg='#ffffff', relief=tk.FLAT)
        telem_card.pack(fill=tk.X, pady=(0, 20))

        tk.Label(telem_card, text="Telemetry", bg='#ffffff', fg='#2c2c2c',
                 font=('Helvetica', 13, 'bold')).pack(anchor=tk.W, padx=20, pady=(15, 5))

        self.alt_label = tk.Label(telem_card, text="Altitude: 0.0 m",
                                  bg='#ffffff', fg='#666666', font=('Helvetica', 11))
        self.alt_label.pack(anchor=tk.W, padx=20, pady=2)

        self.bat_label = tk.Label(telem_card, text="Battery: 100%",
                                  bg='#ffffff', fg='#666666', font=('Helvetica', 11))
        self.bat_label.pack(anchor=tk.W, padx=20, pady=(2, 15))

        # Buttons
        button_frame = tk.Frame(right_panel, bg='#f5f5f5')
        button_frame.pack(fill=tk.X, pady=10)

        self.session_btn = RoundedButton(button_frame, "Start Flight Session", self.toggle_session,
                                         bg_color='#2c2c2c', fg_color='#ffffff',
                                         hover_color='#3c3c3c', width=260, height=50)
        self.session_btn.pack(pady=5)

        RoundedButton(button_frame, "Clear Alerts", self.clear_alerts,
                      bg_color='#666666', fg_color='#ffffff',
                      hover_color='#777777', width=260, height=50).pack(pady=5)

    def connect_radiomaster(self):
        try:
            self.serial_port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            self.connection_status.config(text="Connection: Active", fg='#00cc66')
        except:
            self.connection_status.config(text="Connection: Offline", fg='#ff4444')

    def update_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            pilot_present, confidence, face_bbox, face_detected = self.pilot_detector.detect_pilot_presence(frame)

            # Update pilot presence status
            prev_status = self.is_authorized
            self.is_authorized = pilot_present

            # Show alert if pilot leaves while session active
            if not pilot_present and self.is_armed:
                self.root.after(0, self.show_alert)
            else:
                self.root.after(0, self.hide_alert)

            # Draw face indicator
            if face_bbox:
                x, y, w, h = face_bbox
                color = (100, 200, 100) if pilot_present else (200, 100, 100)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                status = "Pilot Detected" if pilot_present else "No Pilot"
                cv2.putText(frame, status, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Show CNN confidence
            conf_text = f"CNN Confidence: {confidence * 100:.1f}%"
            conf_color = (100, 200, 100) if pilot_present else (200, 100, 100)
            cv2.putText(frame, conf_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

            # Convert for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((620, 465))
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.root.after(0, self.update_ui)

            time.sleep(0.03)

    def update_ui(self):
        # Status indicator dot
        color = '#00cc66' if self.is_authorized else '#ff4444'
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 18, 18, fill=color, outline='')

        # Pilot status
        if self.is_authorized:
            self.pilot_status_label.config(text="● Pilot Present", fg='#00cc66')
            self.pilot_name.config(text="Operator at controls", fg='#2c2c2c')
            self.auth_status.config(text="✓ Pilot Detected at Station", fg='#00cc66')
        else:
            self.pilot_status_label.config(text="● No Pilot", fg='#ff4444')
            self.pilot_name.config(text="Station unattended", fg='#999999')
            self.auth_status.config(text="Waiting for pilot...", fg='#666666')

        # Session status
        if self.is_armed:
            if self.is_authorized:
                self.session_label.config(text="Session: Active", fg='#00cc66')
            else:
                self.session_label.config(text="Session: PILOT ABSENT!", fg='#ff4444')
        else:
            self.session_label.config(text="Session: Inactive", fg='#999999')

        # Telemetry
        self.alt_label.config(text=f"Altitude: {self.drone.altitude:.1f} m")
        self.bat_label.config(text=f"Battery: {self.drone.battery:.0f}%")

    def show_alert(self):
        self.alert_banner.pack(fill=tk.X, after=self.root.children[list(self.root.children.keys())[0]])

    def hide_alert(self):
        self.alert_banner.pack_forget()

    def clear_alerts(self):
        self.hide_alert()

    def control_loop(self):
        while self.running:
            self.drone.update(0.5, 0.5, 0.5, self.is_armed)

            if self.serial_port and self.is_armed and self.is_authorized:
                try:
                    scale = lambda v: int(
                        CRSF_CHANNEL_VALUE_MIN + (v * (CRSF_CHANNEL_VALUE_MAX - CRSF_CHANNEL_VALUE_MIN)))
                    self.crsf_packet.set_channel(0, scale(0.5))
                    self.crsf_packet.set_channel(1, scale(0.5))
                    self.crsf_packet.set_channel(2, scale(0.5))
                    self.crsf_packet.set_channel(3, scale(0.5))
                    self.serial_port.write(self.crsf_packet.build_packet())
                except:
                    pass
            time.sleep(0.02)

    def toggle_session(self):
        self.is_armed = not self.is_armed
        if self.is_armed:
            status = "authorized" if self.is_authorized else "UNAUTHORIZED"
            print(f"Flight session started - Pilot: {status}")
        else:
            print("Flight session ended")

    def emergency_stop(self):
        self.is_armed = False
        self.drone.altitude = 0.0

    def on_close(self):
        self.running = False
        if self.serial_port:
            self.serial_port.close()
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DroneControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()