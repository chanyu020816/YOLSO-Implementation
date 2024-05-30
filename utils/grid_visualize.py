import os
import cv2
import numpy as np
from ultralytics import YOLO

class GridVisualize:
    def __init__(self, model_path: str, grid_size: int, conf: float = 0.5):
        self.model = YOLO(model_path)
        self.grid_size = grid_size
        self.confidence_threshold = conf
        self.colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
            (128, 0, 0),    # 深红色
            (0, 128, 0)     # 深绿色
        ]


    def predict(self, image_path: str):
        pred = self.model.predict(image_path)
        return pred

    def plo_pred(self, image_path: str):
        image = cv2.imread(image_path)
        pred = self.predict(image_path)

        for p in pred:
            for bbox in p.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                class_id = int(class_id)
                color = self.colors[class_id]
                if score < self.confidence_threshold: continue

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                label = f'{score:.2f}'
                cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        file_name = os.path.basename(image_path)[:-4]
        cv2.imwrite(f'./{file_name}_pred.jpg', image)

    def plot_gridpred(self, image_path: str):
        image = cv2.imread(image_path)
        width, height = image.shape[0], image.shape[1]
        pred = self.predict(image_path)
        grid_with_symbol = []

        for p in pred:
            for bbox in p.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                class_id = int(class_id)
                color = self.colors[class_id]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                center_grid = (center[0] // self.grid_size, center[1] // self.grid_size)

                if center_grid not in grid_with_symbol:
                    grid_with_symbol.append(center_grid)
                    i, j = center_grid

                    grid_coord_tl = (i * self.grid_size, j * self.grid_size)
                    grid_coord_br = ((i + 1) * self.grid_size, (j + 1) * self.grid_size)
                    image = self.draw_rectangle(image, grid_coord_tl, grid_coord_br, color, 0.5)
                else:
                    pass

        file_name = os.path.basename(image_path)[:-4]
        cv2.imwrite(f'./{file_name}_grid_pred.jpg', image)


    @staticmethod
    def draw_rectangle(image, top_left, bottom_right, color, alpha):
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

if __name__ == '__main__':
    visualizer = GridVisualize("../train8/weights/best.pt", gird_size = 25)
    visualizer.plot_pred("../data/images/JM20K_1904_025_h3_w2_h0_w0.jpg")
    visualizer.plot_gridpred("../data/images/JM20K_1904_025_h3_w2_h0_w0.jpg")