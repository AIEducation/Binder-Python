from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

img_path = 'photos/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
model = ResNet50(weights='imagenet')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

plt.imshow(img)
print('Распознаное фото:', decode_predictions(preds, top=1)[0][0][1])
