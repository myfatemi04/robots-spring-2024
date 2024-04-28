import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from detect_objects_clipseg import detect_objects_clipseg
from generate_object_candidates import detect as detect_owlv2, draw_set_of_marks

# given an image and a query, write what each object detector detectors
path = 'plan_logs/2024-04-27T01_58_41/images/1.png'
query = 'piece of candy'
# print("Test the object detectors with a given prompt. If you want to reuse the previous prompt or image path, leave the field blank.")
while True:
    # path = input("Image path: ") or path
    image = PIL.Image.open(path)

    # query = input("Query: ") or query

    # heatmap_clipseg = detect_objects_clipseg(image, query)
    # plt.title("CLIPSeg result")
    # plt.imshow(image)
    # plt.imshow(heatmap_clipseg[0], alpha=1/(1+np.exp(-heatmap_clipseg[0])))
    # plt.show()

    dets_ov2 = detect_owlv2(image, query)
    annotated_image = draw_set_of_marks(image, dets_ov2)
    plt.title("OwlV2 result")
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

    break




