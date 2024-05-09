import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer
from mltu.configs import BaseModelConfigs

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, configs_path: str, weights_path: str, model_path: str, *args, **kwargs):
        super().__init__(model_path=model_path, *args, **kwargs)

        # Load configurations from YAML file
        configs = BaseModelConfigs.load(configs_path)
        self.char_list = configs.vocab

        # Load the weights from the last saved checkpoint
        self.load_weights(weights_path)


    def load_weights(self, weights_path: str):
        # Assuming inferenceModel is already defined with the same architecture as the trained model
        # Load the weights into inferenceModel
        self.load_weights(weights_path)

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import tensorflow as tf
    import tf2onnx
    import onnx

    # Convert Keras model to ONNX
    model_path_keras = "/Users/aaronshields/Desktop/CSCI-360/Research_project/Models/04_sentence_recognition/Checkpoints/model.keras"
    model_onnx_path = "/Users/aaronshields/Desktop/CSCI-360/Research_project/Models/OnnxModel/model.onnx"
    model = tf.keras.models.load_model(model_path_keras, custom_objects={'Lambda': tf.keras.layers.Lambda}, safe_mode=False)
    input_signature = [tf.TensorSpec(shape=[None] + list(model.input_shape[1:]), dtype=model.input.dtype)]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, model_onnx_path)

    # Provide paths to the configurations YAML file, weights checkpoint, and ONNX model path
    configs_path = "/Users/aaronshields/Desktop/CSCI-360/Research_project/Models/04_sentence_recognition/202404220023/configs.yaml"
    weights_path = "/Users/aaronshields/Desktop/CSCI-360/Research_project/Models/04_sentence_recognition/Checkpoints/model.keras"
    model_path = model_onnx_path

    # Load the ONNX model
    model = ImageToWordModel(configs_path=configs_path, weights_path=weights_path, model_path=model_path)

    # Run inference
    df = pd.read_csv("Models/04_sentence_recognition/202301131202/val.csv").values.tolist()
    accum_cer, accum_wer = [], []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)
        prediction_text = model.predict(image)
        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print("Image: ", image_path)
        print("Label:", label)
        print("Prediction: ", prediction_text)
        print(f"CER: {cer}; WER: {wer}")
        accum_cer.append(cer)
        accum_wer.append(wer)
        cv2.imshow(prediction_text, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
