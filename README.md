# alan_imuv
part of imuv project, using low frequency clean signal to enhance wide band

##  inference
```bash
export PYTHONPATH=<working dir>/alan_imuv
cd <working direcoty>/alan_imuv/models
```

#### imuv_tasnet_sisnr.py contains a test() function showing the basic setup to load and inference the model, the comment should be clear there for integration into a notebook
```python 
python imuv_tasnet_sisnr.py 
```

#### imuv_crn_sisnr.py contains a test() function showing the basic setup to load and inference the model, the comment should be clear there for integration into a notebook
```python 
python imuv_crn_sisnr.py 
```
