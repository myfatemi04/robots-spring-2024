import generate_object_candidates
import PIL.Image as Image
import matplotlib.pyplot as plt

image = Image.open("droid_sample.png").convert("RGB")

preds = generate_object_candidates.detect(image, "pan")
generate_object_candidates.draw_set_of_marks(image, preds, live=True)
plt.show()
