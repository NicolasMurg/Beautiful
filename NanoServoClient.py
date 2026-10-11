import socket
import time

# Setup Client Configuration
PI_IP = '192.168.2.11'  # Raspberry Pi's static IP
PORT = 65432            # Must match the Pi's port

print("Connecting to Raspberry Pi...")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((PI_IP, PORT))
    print("Connected! Sending servo commands...")
    
    # Example loop: Sweep the servo back and forth
    try:
        while True:
            # Move to 0 degrees
            print("Sending 0°")
            s.sendall(b"0")
            time.sleep(2)
            
            # Move to 90 degrees (Center)
            print("Sending 90°")
            s.sendall(b"90")
            time.sleep(2)
            
            # Move to 180 degrees
            print("Sending 180°")
            s.sendall(b"180")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping script.")
