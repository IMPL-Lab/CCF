pip install torch==2.7 torchvision --index-url https://download.pytorch.org/whl/cu128

pip install setuptools==75.1.0 ninja

git submodule update --init --recursive

cd thirdparty
cd mmcv_torch2
MMCV_WITH_OPS=1 pip install -e .
cd ..

pip install mmsegmentation==0.29.1

cd mmdetection_ccf
pip install -e . --no-build-isolation

cd ..

pip install git+https://github.com/Abyssaledge/TorchEx.git --no-build-isolation
pip install torch-scatter==2.1.2 fvcore spconv-cu120

cd mmdetection3d_ccf
pip install -e .
cd ..

cd nuscenes-devkit_ccf/setup
pip install -e .
cd ../../

# other dependencies
pip install wandb ipdb einops
# Keep NumPy below 2.0 for this OpenMMLab stack. If pip resolves newer binary packages after installing the OpenMMLab submodules, pin the compatible wheels:
pip install yapf==0.40.1 numpy==1.23.5 networkx==2.8.4 motmetrics
pip install opencv-python==4.8.1.78 plyfile==1.1.3 huggingface_hub

# install flash attention
# For a clean Python 3.10 environment with the CUDA 12.8 PyTorch wheels, the tested FlashAttention install is:
pip install flash-attn==2.7.4.post1 --no-build-isolation

# download weights and put to checkpoints folder

# prepare data (including splits)