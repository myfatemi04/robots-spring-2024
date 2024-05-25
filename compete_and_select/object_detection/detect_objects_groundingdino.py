import torch
import PIL.Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from compete_and_select.object_detection.object_detection_utils import add_object_clip_embeddings, draw_set_of_marks

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def detect(image: PIL.Image.Image, label: str):
    inputs = processor(images=image, text=label, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]

    return add_object_clip_embeddings(image, [
        {
            'box': {
                'xmin': results['boxes'][i, 0].item(),
                'ymin': results['boxes'][i, 1].item(),
                'xmax': results['boxes'][i, 2].item(),
                'ymax': results['boxes'][i, 3].item(),
            },
            'score': results['scores'][i].item(),
            'label': results['labels'][i],
        }
        for i in range(results['boxes'].shape[0])
    ])

if __name__ == '__main__':
    from detect_objects import detect as detect_owlv2
    import matplotlib.pyplot as plt
    image = PIL.Image.open("sample_images/1.png")
    dets = detect_owlv2(image, "piece of candy")
    marks = draw_set_of_marks(image, dets)
    plt.axis('off')
    plt.imshow(marks)
    plt.show()
