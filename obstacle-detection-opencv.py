import cv2
import numpy as np
from scipy.spatial import KDTree



net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_obstacles(frame):
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_obstacles = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            detected_obstacles.append([x, y, w, h])

    return frame, detected_obstacles


def find_path(obstacles, start, goal):
    points = [(x + w // 2, y + h // 2) for x, y, w, h in obstacles]
    if not points:
        return [start, goal]

    tree = KDTree(points)
    _, idx_start = tree.query(start)
    _, idx_goal = tree.query(goal)

    return [start, points[idx_start], points[idx_goal], goal]


def select_camera():
    print("Available cameras:")
    print("1. Camera 0")
    print("2. Camera 1")
    choice = input("Enter the camera number (0 or 1): ")

    if choice == '0':
        return cv2.VideoCapture(0)
    elif choice == '1':
        return cv2.VideoCapture(1)
    else:
        print("Invalid choice. Exiting...")
        return None

def main():
    # Select the camera
    cap = select_camera()
    if cap is None or not cap.isOpened():
        print("Camera not available. Exiting...")
        return

   
    window_width, window_height = 1920 , 1080
    cv2.namedWindow('Advanced Obstacle Detection and Path Planning', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Advanced Obstacle Detection and Path Planning', window_width, window_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        processed_frame, obstacles = detect_obstacles(frame)

        
        start = (0, 0)  
        goal = (700, 700)   
        path = find_path(obstacles, start, goal)

        
        for i in range(len(path) - 1):
            cv2.line(processed_frame, path[i], path[i + 1], (255, 0, 0), 2)

        cv2.imshow('Advanced Obstacle Detection and Path Planning', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
