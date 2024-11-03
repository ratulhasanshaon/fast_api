from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
import cv2
import time
from pathlib import Path

app = FastAPI()
capture_images = False  # Control variable to start/stop capturing
output_dir = Path("./captured_images")
output_dir.mkdir(exist_ok=True)

# Background subtraction for motion detection
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def capture_motion():
    global capture_images
    print("Starting motion capture process...")
    
    camera = cv2.VideoCapture(0)  # Re-initialize camera each time the function is called
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
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
        motion_detected = False  # To track if any motion was detected in this frame
        for contour in contours:
            area = cv2.contourArea(contour)
            print(f"Contour area detected: {area}")  # Diagnostic message
            if area > 100:  # Lower threshold for sensitivity
                timestamp = int(time.time())
                img_path = output_dir / f"capture_{timestamp}.jpg"
                cv2.imwrite(str(img_path), frame)  # Save the frame as an image
                print(f"Motion detected! Image saved at {img_path}")
                motion_detected = True
                time.sleep(2)  # Delay to avoid multiple captures for the same motion
                break
        if not motion_detected:
            print("No motion detected in this frame.")
        
    camera.release()
    print("Stopped motion capture process.")


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
