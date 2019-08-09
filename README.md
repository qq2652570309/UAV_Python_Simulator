# UAV_POSTPROCESS

# installation
pip install -r requirement

# generate training data and ground truth
python simulator.py

# preprocessing data
python dataPreprocess.py

# training
python model.py

# generate image
python generateImage.py

# Official tensorflow-gpu binaries are built with: 
cuda 9.0, cudnn 7 since TF 1.5; cuda 10.0, cudnn 7 since TF 1.13. 
