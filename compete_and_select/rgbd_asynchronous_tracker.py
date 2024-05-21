import sys
import time
from threading import Lock, Thread, Event

import PIL.Image
import sam
import torch
from rgbd import RGBD
from torchvision.transforms.functional import to_tensor

sys.path.append("../../Cutie")

# Note that you may need to install hydra-core to do this.
from cutie.inference.inference_core import InferenceCore  # type: ignore
from cutie.utils.get_default_model import get_default_model  # type: ignore

sys.path.pop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RGBDAsynchronousTracker:
    def __init__(self, rgbd: RGBD):
        self.rgbd = rgbd
        self.prev_rgbs = None
        self.prev_pcds = None
        self.prev_object_tracking_mask = None
        self.prev_capture_timestamp = None
        self.thread = Thread(target=self._run)
        self.running = False
        self.memory_queue = []
        self.object_id_counter = 0
        self.memory_queue_lock = Lock()

        # obtain the Cutie model with default parameters -- skipping hydra configuration
        self.cutie = get_default_model()
        # Typically, use one InferenceCore per video
        self.cutie_processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)

        self.publish_event = Event()
        self.tracking_anything = False

    def open(self):
        self.running = True
        self.thread.start()

    # TODO -- def pause(self): pass

    def close(self):
        self.running = False
        self.rgbd.close()
        self.thread.join()

    def create_joint_mask(self, rgb, bounding_boxes, object_ids):
        # Create instance segmentation-like mask.
        masks = sam.boxes_to_masks(rgb, bounding_boxes)
        overall_mask = torch.zeros_like(masks[0], device=device)
        for object_id, mask in zip(object_ids, masks):
            overall_mask[mask != 0] = object_id

    def next(self):
        self.publish_event.wait()
        self.publish_event.clear()
        return self.prev_rgbs, self.prev_pcds, self.prev_object_tracking_mask

    def _run(self):
        with torch.no_grad():
            while self.running:
                try:
                    (rgbs, pcds) = self.rgbd.capture()
                    img_pil = PIL.Image.fromarray(rgbs[0])
                    img = to_tensor(img_pil).cuda().float()

                    if len(self.memory_queue) > 0:
                        # Ensure that batch-adds form an atomic operation.
                        self.memory_queue_lock.acquire()
                        object_ids, bounding_boxes = zip(*self.memory_queue)
                        self.memory_queue = []
                        self.memory_queue_lock.release()
                        
                        overall_mask = self.create_joint_mask(rgbs[0], bounding_boxes, object_ids)
                        output_prob = self.cutie_processor.step(img, overall_mask, objects=object_ids)
                        self.tracking_anything = True
                    elif self.tracking_anything:
                        output_prob = self.cutie_processor.step(img)

                    # Save GPU memory.
                    del img
                    
                    # convert output probabilities to an object mask
                    if self.tracking_anything:
                        mask = self.cutie_processor.output_prob_to_mask(output_prob)
                    else:
                        mask = None

                    self.prev_capture_timestamp = time.time()
                    self.prev_rgbs = rgbs
                    self.prev_pcds = pcds
                    self.prev_object_tracking_mask = mask
                    self.publish_event.set()
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(1)

    def store(self, bounding_boxes):
        # store objects using bounding boxes.
        # adds to a queue which is claimed during the next cycle.
        # returns the object id.
        self.memory_queue_lock.acquire()
        object_ids = []
        for bounding_box in bounding_boxes:
            self.object_id_counter += 1
            object_ids.append(self.object_id_counter)
            self.memory_queue.append((self.object_id_counter, bounding_box))
        self.memory_queue_lock.release()
        return object_ids
        
    def forget(self, object_id: int):
        pass
