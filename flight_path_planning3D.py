import cv2
import numpy as np
from scipy.spatial import KDTree

class Drone3DNavigation:
    def __init__(self, camera_indices):
        self.cameras = [cv2.VideoCapture(idx) for idx in camera_indices]
        self.current_camera = 0
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.flight_relevant_classes = ['person', 'bicycle', 'car', 'airplane', 'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'tree', 'building']

    def switch_camera(self):
        self.current_camera = (self.current_camera + 1) % len(self.cameras)

    def detect_obstacles(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] in self.flight_relevant_classes:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        obstacles = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                # Assuming the obstacle is a cuboid with depth equal to its width
                obstacles.append((x, y, 0, w, h, w, label))  # (x, y, z, width, height, depth, label)

        return frame, obstacles

    def plan_3d_path(self, start, goal, obstacles, space_dimensions):
        def distance(a, b):
            return np.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

        def is_valid(point):
            x, y, z = point
            if not (0 <= x < space_dimensions[0] and 0 <= y < space_dimensions[1] and 0 <= z < space_dimensions[2]):
                return False
            for ox, oy, oz, w, h, d, _ in obstacles:
                if ox <= x < ox + w and oy <= y < oy + h and oz <= z < oz + d:
                    return False
            return True

        def get_neighbors(point):
            x, y, z = point
            neighbors = []
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                new_point = (x + dx, y + dy, z + dz)
                if is_valid(new_point):
                    neighbors.append(new_point)
            return neighbors

        def a_star(start, goal):
            open_set = {start}
            came_from = {}
            g_score = {start: 0}
            f_score = {start: distance(start, goal)}

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
                for neighbor in get_neighbors(current):
                    tentative_g_score = g_score[current] + distance(current, neighbor)
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + distance(neighbor, goal)
                        if neighbor not in open_set:
                            open_set.add(neighbor)

            return None

        return a_star(start, goal)

    def visualize_3d_path(self, frame, path):
        if path:
            for i in range(len(path) - 1):
                start = (int(path[i][0]), int(path[i][1]))
                end = (int(path[i+1][0]), int(path[i+1][1]))
                cv2.line(frame, start, end, (0, 0, 255), 2)
                # Visualize altitude change
                cv2.putText(frame, f"Alt: {path[i][2]}", start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return frame

    def run(self):
        start = (50, 50, 50)  # Example start point (x, y, z)
        goal = (500, 400, 100)  # Example goal point (x, y, z)
        space_dimensions = (640, 480, 200)  # Example 3D space dimensions

        while True:
            ret, frame = self.cameras[self.current_camera].read()
            if not ret:
                print("Failed to capture frame")
                break

            processed_frame, obstacles = self.detect_obstacles(frame)
            path = self.plan_3d_path(start, goal, obstacles, space_dimensions)
            visualized_frame = self.visualize_3d_path(processed_frame, path)

            cv2.imshow('3D Flight Path Planning', visualized_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.switch_camera()

        for camera in self.cameras:
            camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage with two cameras (indices 0 and 1)
    drone_nav = Drone3DNavigation([1])
    drone_nav.run()