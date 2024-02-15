mamba install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
# We select specific CUDA version (11.8)
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd hand_object_detector
mamba install cffi pytorch::torchvision
cd lib
python setup.py build develop
cd ../..
