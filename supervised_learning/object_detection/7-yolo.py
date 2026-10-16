#!/usr/bin/env python3
"""Module that defines the Yolo class for object detection"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo detector

        model_path: path to the Darknet Keras model
        classes_path: path to the list of class names
        class_t: box score threshold for the initial filtering step
        nms_t: IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes the Darknet model outputs into boundary boxes,
        box confidences and class probabilities
        """
        image_h, image_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_h, grid_w, _, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            b_x = (self.sigmoid(t_x) + c_x) / grid_w
            b_y = (self.sigmoid(t_y) + c_y) / grid_h
            b_w = p_w * np.exp(t_w) / input_w
            b_h = p_h * np.exp(t_h) / input_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = (b_x - b_w / 2) * image_w
            box[..., 1] = (b_y - b_h / 2) * image_h
            box[..., 2] = (b_x + b_w / 2) * image_w
            box[..., 3] = (b_y + b_h / 2) * image_h

            boxes.append(box)
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters out boxes whose best class score is below class_t
        Returns (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes, box_classes, box_scores = [], [], []

        for box, conf, probs in zip(boxes, box_confidences,
                                    box_class_probs):
            scores = conf * probs
            classes = np.argmax(scores, axis=-1)
            best_scores = np.max(scores, axis=-1)

            mask = best_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(best_scores[mask])

        return (np.concatenate(filtered_boxes),
                np.concatenate(box_classes),
                np.concatenate(box_scores))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Removes overlapping boxes of the same class (IoU > nms_t)
        Returns (box_predictions, predicted_box_classes,
        predicted_box_scores) ordered by class and box score
        """
        box_predictions, predicted_classes, predicted_scores = [], [], []

        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            order = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_classes.append(cls)
                predicted_scores.append(cls_scores[0])

                iou = self.iou(cls_boxes[0], cls_boxes[1:])

                keep = iou <= self.nms_t
                cls_boxes = cls_boxes[1:][keep]
                cls_scores = cls_scores[1:][keep]

        return (np.array(box_predictions),
                np.array(predicted_classes),
                np.array(predicted_scores))

    @staticmethod
    def iou(box, boxes):
        """Intersection over Union between one box and several boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return inter / (area_box + area_boxes - inter)

    @staticmethod
    def load_images(folder_path):
        """
        Loads all the images of a folder
        Returns (images, image_paths)
        """
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        images = [cv2.imread(path) for path in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes the images to the model input size and rescales
        their pixel values to [0, 1]
        Returns (pimages, image_shapes)
        """
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        pimages, image_shapes = [], []
        for image in images:
            image_shapes.append(image.shape[:2])
            # attention: cv2.resize prend (largeur, hauteur)
            resized = cv2.resize(image, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with its boundary boxes, class names and
        box scores; saves it in detections/ if the s key is pressed
        """
        for box, cls, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = '{} {:.2f}'.format(self.class_names[cls], score)
            cv2.putText(image, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            os.makedirs('detections', exist_ok=True)
            cv2.imwrite(os.path.join('detections', file_name), image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Runs object detection on all the images of a folder and
        displays them with their boxes
        Returns (predictions, image_paths)
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        predictions = []
        for i, image in enumerate(images):
            image_outputs = [output[i] for output in outputs]
            boxes, confidences, class_probs = self.process_outputs(
                image_outputs, image_shapes[i])
            boxes, classes, scores = self.filter_boxes(
                boxes, confidences, class_probs)
            boxes, classes, scores = self.non_max_suppression(
                boxes, classes, scores)
            predictions.append((boxes, classes, scores))

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(image, boxes, classes, scores, file_name)

        return predictions, image_paths
