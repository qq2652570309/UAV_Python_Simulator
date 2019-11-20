# UAV_Python_Simulator


## fushion
* tasklist: data_tasks (3k, 60, 15, 5)
* init:     data_init  (3k, 100, 100)
* subnet output: data_subnet   (3k, 60, 100, 100)
* nofly: data_env (3k, 100, 100)
* mainnet label: label_mainnet (3k, 100, 100)

## subnet
* tasklist: data_tasks   (180k, 15, 5)
* nofly:  data_env (180k, 100, 100)
* label:    label_subnet (180k, 100, 100)

## installation
pip install -r requirement

## generate whole data
```
python dataPreprocess.py
```

## generate data for Main Network
simulator_mainNet.py

## generate data for Main Network
simulator_subNet.py

## preprocessing data
dataPreprocess.py

## generate image
generateImage.py

## redunce warming
pip install "numpy<1.17" or numpy==1.16.4
