import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from PIL import Image
import cv2

focused_score = [1, 1, 2, 3, 4, 5, 6, 7, 8]
plt.figure(figsize=(2, 2), dpi=400)
plt.plot([time for time in range(len(focused_score))], focused_score)
plt.savefig("focused_score.png")

image = np.array(Image.open("focused_score.png").resize((640, 640)))
image = Image.fromarray(image)
buff = BytesIO()
image.save(buff, format="png")
base64_image = base64.b64encode(buff.getvalue()).decode("utf-8")