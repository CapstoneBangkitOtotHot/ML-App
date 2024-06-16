from enum import Enum
from typing import Any
import cv2
from ultralytics import YOLO
import pprint
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    from encode import ndarray_to_base64
    from metadata import *
else:
    from .encode import ndarray_to_base64
    from .metadata import *
class FreshFruitNotSupportedException(Exception):
    """Raised when some fruits cannot be analyzed it's freshness"""
    pass

from inspect import getsourcefile
from os.path import abspath, dirname, join

base_path = dirname(abspath(getsourcefile(lambda:0)))

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
        result = self.model_classification.predict(image, conf=0.5)[0]

        res = []

        # TODO: OpenCV cut to bounding box - DONE
        # TODO: Sort by confidence and only get the highest confidence - CHANGED
        # TODO: Feed yolo output to matching freshness model - DONE
        boxes = result.boxes

        # Iterate through each box
        for i in range(len(boxes)):
            # Each box contains (x1, y1, x2, y2, confidence, class)
            x1, y1, x2, y2 = boxes.xyxy[i]
            conf = float(boxes.conf[i].item())
            cls = int(boxes.cls[i].item())
            fruit_class = FruitClass(cls)
            # Convert to integer
            x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])

            # Crop the image using the bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]

            print(model_regressions.keys())

            model_regression = model_regressions[fruit_class]

            freshness_percentage = model_regression(cropped_image)
            freshness_days = None

            if freshness_percentage:
                freshness_percentage = float(freshness_percentage[0, 0])

                fruit_metadata = FruitMetadata[fruit_class]
                freshness_days = freshness_percentage * (fruit_metadata['range']/2)
            

            res.append(
                {
                    # ada jika classification berhasil
                    "fruit_class" : cls,
                    "fruit_class_string" : FruitMetadata[fruit_class]['string'],
                    "cropped_img": cropped_image,
                    "confidence" : conf,

                    "freshness_percentage"  : freshness_percentage,
                    "freshness_days"  : freshness_days,
                }
            )

        return {
            "orig_img"   : image,

            "inferences": res,
        }



class ClassificationModel(YOLO):

    def __init__(self, **kwargs) -> None:
        import inspect, os.path
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path     = os.path.dirname(os.path.abspath(filename))
        super().__init__(path + '/models/classification_model.pt')

    # Future custom intermediate function
    

@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):
    # Customized Mean Squared Error (MSE) to reflect the range of labels
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Custom metric
@tf.keras.utils.register_keras_serializable()
def custom_metric(y_true, y_pred):
    # Customized Mean Absolute Error (MAE) to reflect the range of labels
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.minimum(diff, 20))
                 

class RegressionModel():
    def __init__(self, model_path) -> None:
        print(model_path)
        self.model_path = model_path
        if (model_path != ''):
            self.model = tf.keras.models.load_model(
                            model_path,  
                            custom_objects={
                                'custom_loss': custom_loss, 
                                'custom_metric': custom_metric
            })
        else:
            self.model = None

    def preprocess_image(self, image):
        resized_image = cv2.resize(image, (360, 360))
        normalized_image = resized_image/255
        add_none_dim = np.expand_dims(normalized_image, axis=0)
        return add_none_dim

    def __call__(self, image) -> Any:
        print(self.model_path)
        if self.model:

            image = self.preprocess_image(image)
            return self.model.predict(image)
        
        return None

# Create model for each fruit
model_regressions = {
    FruitClass.TOMATO: RegressionModel(join(base_path, 'models/tomato.keras')),
    FruitClass.MANGO: RegressionModel(join(base_path, 'models/mango.keras')),
    FruitClass.SAPODILLA: RegressionModel(join(base_path, 'models/sapodilla.keras')),
    FruitClass.APPLE: RegressionModel(join(base_path, 'models/apple.keras')),
    FruitClass.PEACH: RegressionModel(''),
    FruitClass.BANANA: RegressionModel(''),
    FruitClass.STRAWBERRY: RegressionModel(''),
}

inference_model = InferenceModel() 


# khusus untuk ML DEBUGGING
# Hanya dieksekusi jika dieksekusi secara langsung lewat terminal
# Jika diimport, lewat import x atau from x import, blok kode ini
# tidak akan dieksekusi
if __name__ == '__main__':
    image_path = 'apel2.jpg'
    image = cv2.imread(image_path)

    print(inference_model(image))
