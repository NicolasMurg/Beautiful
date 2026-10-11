import socket
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

# Setup Server Configuration
HOST = '192.168.2.11'  # Raspberry Pi's static IP
PORT = 65432           # Arbitrary non-privileged port

# Setup Servo on GPIO 18
# Using PiGPIOFactory provides jitter-free hardware-timed PWM
factory = PiGPIOFactory()
servo = Servo(18, pin_factory=factory)

print("Starting Servo Server, waiting for Jetson Nano...")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(Connected by {addr})
        while True:
            data = conn.recv(1024)
            if not data:
                break
            
            # Decode the message from bytes to a string
            message = data.decode('utf-8')
            
            try:
                # Gpiozero expects a value between -1.0 (min) and 1.0 (max)
                # Map standard 0-180 angles to -1.0 to 1.0
                angle = float(message)
                servo_value = (angle / 90.0) - 1.0
                
                # Constrain bounds
                servo_value = max(-1.0, min(1.0, servo_value))
                
                print(f"Moving servo to angle: {angle}° (Value: {servo_value:.2f})")
                servo.value = servo_value
                
            except ValueError:
                print(f"Received invalid data: {message}")
