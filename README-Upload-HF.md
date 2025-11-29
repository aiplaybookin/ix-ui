# Upload Model to Hugging Face Hub

### 1. Install HF
```
pip install huggingface_hub
```

### 2. Login to HF
```
huggingface-cli login
```

You'll need an access token:

1. Go to huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "upload-model")
4. Select "Write" permission
5. Copy the token and paste it when prompted

### 3. Update the configurations
Use your own details in file ```upload_to_hf.py```-
```
HF_USERNAME = "your-username"  # Your actual HF username
MODEL_NAME = "smollm2-135m-raw-trained"  # Whatever you want to name it
CHECKPOINT_PATH = "checkpoints/smollm2-phase2-final.ckpt"  # Your checkpoint path
```

### 4. Run the upload script
```
python upload_to_hf.py
```

### 5. Verify upload
```
https://huggingface.co/your-username/smollm2-135m-raw-trained
```

### 6. Test 

```
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-username/smollm2-135m-finetuned")
model = AutoModelForCausalLM.from_pretrained("your-username/smollm2-135m-finetuned")

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

