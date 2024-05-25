mamba install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
# We select specific CUDA version (11.7 <= 11.8)
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# CUDA 11.7 only allows up to gcc 11
mamba install conda-forge::gxx_linux-64
mamba install conda-forge::gcc=11.4.0
# Later matplotlib versions are unsupported
mamba install conda-forge::matplotlib~=3.7
cd hand_object_detector
mamba install cffi pytorch::torchvision
cd lib
# It is important that `torch.cuda.is_available() == True` during this step.
python setup.py build develop
cd ../..
