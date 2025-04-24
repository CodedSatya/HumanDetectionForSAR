import tensorflow as tf
from tensorflow.keras import Model, layers

class YOLO(Model):
  def __init__(self, num_classes=5, S=7, B=2):
    super(YOLO, self).__init__()
    self.S = S
    self.B = B
    self.num_classes = num_classes
    self.output_dim = B*(5 + num_classes)
    
    self.backbone = tf.keras.Sequential([
        layers.Conv2D(16, 3, padding="same", input_shape=(224, 224, 3)),
        layers.ReLU(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, 3, padding="same"),
        layers.ReLU(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, 3, padding='same'),
        layers.Relu(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, 3, padding='same'),
        layers.ReLU(),
        layers.MaxPooling2D(4, 4),
    ])

    self.head = tf.keras.Sequential([
        layers.Conv2D(self.output_dim, 1),
        layers.Activation('sigmoid')
    ])

  def call(self, inputs, training=False):
    features = self.backbone(inputs)
    prediction = self.head(features)
    return prediction
if __name__  == "__mian__":
  model = YOLO(5, S=7, B=2)
  dummy_input = tf.random.normal((1, 224, 224, 3))
  output = model(dummy_input)
  print(f"Output Shape : {output}")