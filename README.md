# ix-ui

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Watch GPU
```
watch -n 1 nvidia-smi
```

# Run inference/demo
python inference.py

# Upload to Hugging Face
huggingface-cli login
python upload_to_hf.py

# View training logs
tensorboard --logdir=logs/


## Known issues: 
1. Fix validation dataset : There should be a split of data from the file and validatin dataset should be different.
    - Current using a small set of same train set.
2. During validation : GPU is not used properly
