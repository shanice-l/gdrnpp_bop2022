```
pip install nvidia-pyindex
# problem for 8: https://github.com/NVIDIA-AI-IOT/torch2trt/issues/557
pip install nvidia-tensorrt==7.2.3.4

git clone git@github.com:NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python setup.py install

# python setup.py install --plugins  # do not work
```
