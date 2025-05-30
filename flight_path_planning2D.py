import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Flight-relevant object classes
flight_relevant_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

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
            if confidence > 0.5 and classes[class_id] in flight_relevant_classes:
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

    obstacles = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            obstacles.append((x, y, w, h, label))

    return frame, obstacles

def plan_path(frame_shape, obstacles, start, goal):
    height, width = frame_shape[:2]
    grid = np.ones((height, width), dtype=np.uint8) * 255

    # Mark obstacles on the grid
    for x, y, w, h, _ in obstacles:
        cv2.rectangle(grid, (x, y), (x + w, y + h), 0, -1)

    # Dilate obstacles to create a safety margin
    kernel = np.ones((20, 20), np.uint8)
    grid = cv2.dilate(grid, kernel, iterations=1)

    # A* pathfinding
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_neighbors(current):
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1),
                     (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] != 0]

    def a_star(start, goal):
        open_set = set([start])
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score[x])

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + heuristic(current, neighbor)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)

        return None

    path = a_star(start, goal)
    return path

def main():
    cap = cv2.VideoCapture(0)
    start = (50, 50)  # Example start point
    goal = (600, 400)  # Example goal point

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        processed_frame, obstacles = detect_obstacles(frame)

        # Plan path
        path = plan_path(frame.shape, obstacles, start, goal)

        # Draw path
        if path:
            for i in range(len(path) - 1):
                cv2.line(processed_frame, path[i], path[i+1], (0, 0, 255), 2)

        # Draw start and goal
        cv2.circle(processed_frame, start, 10, (255, 0, 0), -1)
        cv2.circle(processed_frame, goal, 10, (0, 255, 0), -1)

        cv2.imshow('Flight Path Planning', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()