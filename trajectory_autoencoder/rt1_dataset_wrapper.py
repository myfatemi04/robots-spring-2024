import torch
import torch.utils.data
# This is the storage format for the RT-1 dataset.
import tensorflow_datasets as tfds


class RT1Dataset(torch.utils.data.Dataset):
    """
    Extremely simple wrapper; just meant to access a couple internal fields more easily.
    """
    def __init__(self, rt1_dataset_folder):
        super().__init__()

        self.rt1_dataset_folder = rt1_dataset_folder
        builder = tfds.builder_from_directory('/scratch/gsk6me/WORLDMODELS/datasets/rt-1-data-release')
        self._dataset = builder.as_data_source()
        
    def __len__(self):
        return len(self._dataset['train'])
    
    def __getitem__(self, index: int):
        # Returns a list of images and a natural language instruction
        steps = self._dataset['train'][index]['steps']
        images = []
        instruction: bytes = steps[0]['observation']['natural_language_instruction']
        instruction = instruction.decode('utf-8')
        # Here, we'll have to choose how we sample which steps to train with.
        for i in range(min(len(steps), 25)):
        # for i in range(len(steps)):
            # image needs to be scaled to the range 0.0-1.0.
            # also, needs to go from (h, w, c) to (c, h, w)
            images.append(torch.tensor(steps[i]['observation']['image'] / 255.0).permute(2, 0, 1))

        return (instruction, images)

