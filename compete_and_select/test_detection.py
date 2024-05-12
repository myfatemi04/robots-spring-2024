import detect_objects
import PIL.Image as Image
import matplotlib.pyplot as plt

image = Image.open("droid_sample.png").convert("RGB")

preds = detect_objects.detect(image, "pan")
detect_objects.draw_set_of_marks(image, preds, live=True)
plt.show()
