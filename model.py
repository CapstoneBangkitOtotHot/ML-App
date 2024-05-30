from enum import Enum
from typing import Any
import cv2
from ultralytics import YOLO

class FruitClass(Enum):
    PEACH      = 1
    PEAR       = 2
    BANANA     = 3
    APPLE      = 4
    STRAWBERRY = 5
    SAPODILLA  = 6
    MANGO      = 7

class InferenceModel():
    """
    A model class to wrap the internal works of ML Team.

    This model includes 2 step end-to-end model of classification
    and regression model

    input: image

    output: list of {label, freshness}



    See implementation to dig deeper
    """
    
    def __init__(self) -> None:
        self.model_classification = ClassificationModel()

    
    def __call__(self, image):
        """
        use the instantiated class as callable function
        e.g.
        model = InferenceModel
        result = model(data)
        """
        results = self.model_classification.predict(image)

        # TODO: OpenCV cut to bounding box - DONE
        # TODO: Sort by confidence and only get the highest confidence - IN PROGRESS
        # TODO: Feed yolo output to matching freshness model - IN PROGRESS
        for result in results:
            # Extract the Boxes object
            boxes = result.boxes

            # Iterate through each box
            for i in range(len(boxes)):
                # Each box contains (x1, y1, x2, y2, confidence, class)
                x1, y1, x2, y2 = boxes.xyxy[i]
                conf = boxes.conf[i]
                cls = boxes.cls[i]
                fruit_class = FruitClass(cls.item())
                # Convert to integer
                x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])

                # Crop the image using the bounding box coordinates
                cropped_image = image[y1:y2, x1:x2]

                model_regression = model_regressions[fruit_class]
                freshness = model_regression(cropped_image)
        return {
            # pasti ada
            "orig_img"   : image,

            # ada jika classification berhasil
            "class"      : cls,
            "cropped_img": cropped_image,
            "confidence" : conf,


            "freshness"  : freshness,
        }



class ClassificationModel(YOLO):

    def __init__(self, **kwargs) -> None:
        import inspect, os.path
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path     = os.path.dirname(os.path.abspath(filename))
        super().__init__(path + '/models/classification_model.pt')

    # Future custom intermediate function


class RegressionModel():
    def __init__(self, model_path) -> None:
        ...

    def __call__(self, image) -> Any:
        return 1

# Create model for each fruit
model_regressions = {
    FruitClass.BANANA: RegressionModel(''),
    FruitClass.MANGO: RegressionModel(''),
    FruitClass.SAPODILLA: RegressionModel(''),
    FruitClass.APPLE: RegressionModel('')
}

inference_model = InferenceModel() 