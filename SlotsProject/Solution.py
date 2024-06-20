import os
import torch
from ultralytics import YOLO
import easyocr
import cv2

class Solution:
    def __init__(self):
        """
        Initialize your data structures here
        """
        #init
        self.model_weights = r'D:\tanishkaka\yebhi.pt'
        self.model = YOLO(self.model_weights)
        self.reader = easyocr.Reader(['en'])
        self.base_path = r'D:\tanishkaka\SlotsProject\Problems'
    
    

    def get_confidences_and_ocr(self, image_path, class_index, requires_ocr=False):
        # inference the model 
        results = self.model(image_path)
        
        # dict to store conf,bbox results
        cls_conf_dict = {}
        cls_bbox_dict = {}

        # fill the dictionary
        for result in results:
            boxes = result.boxes 
            for c, cf, bbox in zip(boxes.cls, boxes.conf, boxes.xywh):
                c = int(c.item()) 
                if c in cls_conf_dict:
                    cls_conf_dict[c].append(cf.item())
                    cls_bbox_dict[c].append(bbox.tolist())
                else:
                    cls_conf_dict[c] = [cf.item()]
                    cls_bbox_dict[c] = [bbox.tolist()]

        # multiplt bboxes->multiple confidence->take avg
        def calculate_average_confidence(cls_conf_dict, key):
            confidences = cls_conf_dict.get(key, [])
            if confidences:
                average_confidence = sum(confidences) / len(confidences)
                return average_confidence
            else:
                print(f"No confidences found for class {key}.")
                return None

        # Get the average confidence for the specified class index
        average_confidence = calculate_average_confidence(cls_conf_dict, class_index)
        
        if requires_ocr and average_confidence is not None and class_index in cls_bbox_dict:
            bbox = cls_bbox_dict[class_index][0]
            image = cv2.imread(image_path)
            
            # Calculate the top-left corner of the bounding box
            x_center, y_center, width, height = bbox
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # cropped image 
            cropped_image = image[y1:y2, x1:x2]
            
            # ocr krdo
            ocr_results = self.reader.readtext(cropped_image)
            
            # text 
            amount = ""
            for res in ocr_results:
                text = res[1]
                if any(char.isdigit() for char in text):  # Check if the text contains any digit
                    amount = text
                    break
            
            return average_confidence, amount
        else:
            return average_confidence, ""

    def get_answer(self, problem):
        """
        ### class_names = {
     0: 'bet', 1: 'bigwin', 
     2: 'bottle', 3: 'bottomrightbet',
     4: 'boy', 5: 'candymancan', 
     6: 'fruit', 7: 'hat', 
     8: 'spin', 9: 'totalwin'}


        Problem contains the name of the set to solve. You can use this to retrieve the images from the set.
        The result should be an array of length 2 with elements of type number (integer / float)
        """
        image_path=os.path.join(self.base_path,problem,'Image.png')
        image_path1 = os.path.join(self.base_path, problem, 'Test1.png')
        image_path2 = os.path.join(self.base_path, problem, 'Test2.png')

        if problem == 'Set1':
            class_index =  next((int(box.cls.item()) for result in self.model(image_path) for box in result.boxes), None) #or mention class index explicitly ,doesnt matter
            confidence1, _ = self.get_confidences_and_ocr(image_path1, class_index)
            confidence2, _ = self.get_confidences_and_ocr(image_path2, class_index)
            return [confidence1, confidence2]
        elif problem == 'Set2':
            class_index = next((int(box.cls.item()) for result in self.model(image_path) for box in result.boxes), None)
            confidence1, _ = self.get_confidences_and_ocr(image_path1, class_index)
            confidence2, _ = self.get_confidences_and_ocr(image_path2, class_index)
            return [confidence1, confidence2]
        elif problem == 'Set3':
            class_index = next((int(box.cls.item()) for result in self.model(image_path) for box in result.boxes), None)
            confidence1, _ = self.get_confidences_and_ocr(image_path1, class_index)
            confidence2, _ = self.get_confidences_and_ocr(image_path2, class_index)
            return [confidence1, confidence2]
        elif problem == 'Set4':
            class_index = next((int(box.cls.item()) for result in self.model(image_path) for box in result.boxes), None)
            confidence1, _ = self.get_confidences_and_ocr(image_path1, class_index)
            confidence2, _ = self.get_confidences_and_ocr(image_path2, class_index)
            return [confidence1, confidence2]
        elif problem == 'Set5':
            class_index = next((int(box.cls.item()) for result in self.model(image_path) for box in result.boxes), None)  
            confidence1, _ = self.get_confidences_and_ocr(image_path1, class_index)
            confidence2, _ = self.get_confidences_and_ocr(image_path2, class_index)
            return [confidence1, confidence2]
        elif problem == 'Set6':
            class_index = next((int(box.cls.item()) for result in self.model(image_path) for box in result.boxes), None)
            confidence1, _ = self.get_confidences_and_ocr(image_path1, class_index)
            confidence2, _ = self.get_confidences_and_ocr(image_path2, class_index)
            return [confidence1, confidence2]
        elif problem == 'Set7':
            class_index =next((int(box.cls.item()) for result in self.model(image_path) for box in result.boxes), None)
            confidence1, _ = self.get_confidences_and_ocr(image_path1, class_index)
            confidence2, _ = self.get_confidences_and_ocr(image_path2, class_index)
            return [confidence1, confidence2]
        elif problem == 'Set8':
            class_index = 9 #cant be precalculated
            _, ocr_result1 = self.get_confidences_and_ocr(image_path1, class_index, requires_ocr=True)
            _, ocr_result2 = self.get_confidences_and_ocr(image_path2, class_index, requires_ocr=True)
            return [ocr_result1, ocr_result2]
        
        else:  # Assume 'Set9'
            class_index = 3  
            _, ocr_result1 = self.get_confidences_and_ocr(image_path1, class_index, requires_ocr=True)
            _, ocr_result2 = self.get_confidences_and_ocr(image_path2, class_index, requires_ocr=True)
            return [ocr_result1, ocr_result2]
