from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import time
from pathlib import Path
import io

app = FastAPI()
capture_images = False
output_dir = Path("./captured_images")
output_dir.mkdir(exist_ok=True)
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Initialize camera
camera = cv2.VideoCapture(0)

# Define motion detection parameters
MIN_CONTOUR_AREA = 5000       # Minimum area of contour to be considered motion
COOLDOWN_PERIOD = 5           # Cooldown period between captures (in seconds)
FRAMES_TO_CONFIRM = 3         # Number of frames to confirm motion

last_capture_time = 0
motion_frames = 0  # Counter for continuous frames with motion

def capture_motion():
    global capture_images, last_capture_time, motion_frames
    print("Starting motion capture process...")
    
    while capture_images:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Find contours to detect motion
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:
                print(f"Detected motion with contour area: {area}")
                motion_detected = True
                break

        # Check if motion is detected across multiple frames
        if motion_detected:
            motion_frames += 1
        else:
            motion_frames = 0  # Reset if no motion in current frame

        # Confirm sustained motion across multiple frames and apply cooldown
        if motion_frames >= FRAMES_TO_CONFIRM:
            current_time = time.time()
            if current_time - last_capture_time > COOLDOWN_PERIOD:
                timestamp = int(current_time)
                img_path = output_dir / f"capture_{timestamp}.jpg"
                cv2.imwrite(str(img_path), frame)
                print(f"Motion confirmed! Image saved at {img_path}")
                last_capture_time = current_time  # Reset cooldown timer
                motion_frames = 0  # Reset motion frame count after capture

        time.sleep(0.1)  # Short delay to reduce CPU usage
    
    print("Stopped motion capture process.")


def video_stream_generator():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Encode the frame as JPEG
        _, jpeg_frame = cv2.imencode(".jpg", frame)
        frame_bytes = jpeg_frame.tobytes()

        # Yield frame as part of an HTTP multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


@app.get("/video-feed")
async def video_feed():
    return StreamingResponse(video_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
async def index():
    html_content = """
    <html>
    <head>
        <title>Camera Feed</title>
    </head>
    <body>
        <h1>Live Camera Feed with Motion Detection</h1>
        <img src="/video-feed" width="640" height="480" />
        <br><br>
        <button onclick="startCapture()">Start Motion Capture</button>
        <button onclick="stopCapture()">Stop Motion Capture</button>
        <script>
            async function startCapture() {
                await fetch('/start-capture');
                alert("Motion capture started.");
            }
            async function stopCapture() {
                await fetch('/stop-capture');
                alert("Motion capture stopped.");
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/start-capture")
async def start_capture(background_tasks: BackgroundTasks):
    global capture_images
    if not capture_images:
        capture_images = True
        background_tasks.add_task(capture_motion)
        return {"status": "Capture started"}
    return {"status": "Already capturing"}


@app.get("/stop-capture")
async def stop_capture():
    global capture_images
    capture_images = False
    return {"status": "Capture stopped"}


@app.get("/captured-images")
async def get_captured_images():
    images = list(output_dir.glob("*.jpg"))
    image_files = [{"filename": img.name} for img in images]
    return {"images": image_files}


@app.get("/captured-images/{filename}")
async def get_image(filename: str):
    img_path = output_dir / filename
    if img_path.exists():
        return FileResponse(img_path, media_type="image/jpeg")
    return {"error": "Image not found"}
