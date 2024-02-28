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
        instruction = steps[0]['observation']['natural_language_instruction']
        for i in range(len(steps)):
            images.append(steps[i]['observation']['image'])

        return (instruction, images)

