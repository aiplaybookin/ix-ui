# Training SmolL2-135M from scratch

## Key Highlights:
### 135M model parameters - Let's see below how -

We are using below hyperparameters for model :

```
hidden_size = 576           # Embedding dimension (d_model)
intermediate_size = 1536    # FFN inner dimension
num_attention_heads = 9     # Query heads
num_key_value_heads = 3     # Key/Value heads (GQA, sharing)
num_hidden_layers = 30      # Number of transformer blocksp
vocab_size = 49152          # Vocabulary size
tie_word_embeddings = True  # i.e. share input/output embeddings
```

#### Simple Diagram to illustrate complete architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    SmolLM2-135M                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  1. EMBEDDING LAYER                                 │   │
│   │     vocab_size × hidden_size (embed dim)            │   │
│   │     49,152 × 576 = 28,311,552                       │   │
│   └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  2. TRANSFORMER BLOCKS (×30)                        │   │
│   │                                                     │   │
│   │     Each block has:                                 │   │
│   │     - Self-Attention                                │   │
│   │     - Feed-Forward Network (MLP)                    │   │
│   │     - Layer Norms                                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  3. FINAL LAYER NORM                                │   │
│   │     hidden_size = 576                               │   │
│   └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  4. LM HEAD (tied with embeddings)                  │   │
│   │     0 (shares weights with embedding)               │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
#### 1. Embedding Layer
a. vocab size is 49,152 distinct tokens (i.e. 49,152 distinct words/ parts of words). Model knows only 49,152 distinct tokens.

b. hidden size (or embedding dimensions) is 576, i.e. every token is coverted to list of 576 numbers or coordinates in space.
```
49,152 (Vocab) × 576 (hidden size/Embedding dimension) = 28,311,552
```
c. This is a static lookup table for prediction of words. Its 28M parameters, ~20% of whole model parameters.

#### 2. Transformer 

#### 2.1. Self attention
a. Model uses ```Grouped Query Attention (GQA)``` and we have ```num attention head = 9```
```
head_dim = hidden_size / num_attention_heads
         = 576 / 9 = 64
```

In simple attention (not GQA) Q, k, V calculation would have been 
```
 576 (hidden size/Embedding dimension) / 9 (num attention head) = 64 (There will be 9 such Q/V heads)

 576(i/p) * 64 * 9 = 331776 # total params
```
But due to GQA, having ```num_key_value_heads = 3``` , 3 Query heads share 1 Key/Value head. 

Instead of having 9 heads for K and V (like Q has), they only have 3 heads.
```
576 (i/p) * (3 heads * 64 dims) = 110592 # total params
```
This makes the K and V matrices 3x smaller than the Q matrix, saving parameters.

```
Q projection: hidden_size × hidden_size
            = 576 × 576 = 331,776

K projection: hidden_size × (num_key_value_heads × head_dim)
            = 576 × (3 × 64) = 576 × 192 = 110,592

V projection: hidden_size × (num_key_value_heads × head_dim)  
            = 576 × 192 = 110,592

O projection: hidden_size × hidden_size
            = 576 × 576 = 331,776

Attention Total = 331,776 + 110,592 + 110,592 + 331,776
                = 884,736
```

**Analogy**

The difference lies in how we resource the team. In this model, we have 9 Analysts (Heads).

**The Old Way (Simple Multi-Head Attention)**
In a standard model (like GPT-2), every Analyst gets their own private filing cabinet.

* 9 Analysts (Queries) need 9 Filing Cabinets (Keys/Values).

* This is expensive! Storing 9 sets of filing cabinets takes up a lot of room (parameters and memory).

**The New Way (Grouped Query Attention - GQA)**

This is what SmolLM2 uses. We realize that while Analysts ask different questions, they often look in the same places. So, we force them to share.

* We put the 9 Analysts into 3 Groups (3 Analysts per group).

* Each group shares 1 Filing Cabinet.

* Total: 9 Analysts (Queries) but only 3 Filing Cabinets (Keys/Values).

#### 2.2. Feed-Forward Network (MLP)

- MLP are ususally 2/3rd of parameters in transformer blocks. We know we took ```intermediate_size = 1536```, FFN inner dimension.

- Inspired from LLaMA it uses SwiGLU, which has 3 linear layers. 

- This expands the data from 576 to 1,536 (Intermediate Size), processes it, and squeezes it back to 576.

```
gate_proj: hidden_size × intermediate_size
         = 576 × 1,536 = 884,736

up_proj:   hidden_size × intermediate_size
         = 576 × 1,536 = 884,736

down_proj: intermediate_size × hidden_size
         = 1,536 × 576 = 884,736

MLP Total = 884,736 × 3 = 2,654,208
```

#### 2.3. Layer Norms (RMSNorm - weights only, no bias)
- These are tiny learnable numbers (RMSNorm) that act like volume knobs, keeping the numbers stable so they don't explode (get too high) or vanish (get to zero) as they pass through the network. There is one parameter for every hidden dimension (576) at the start and end of the block.
```
input_layernorm:          576
post_attention_layernorm: 576

LayerNorm Total = 1,152

```
#### 2.4 In total (Stacking 30 layers)
Total Per Block
```
Attention + MLP + LayerNorms
= 884,736 + 2,654,208 + 1,152
= 3,540,096 per block
```
All 30 Transformer Blocks
```
30 × 3,540,096 = 106,202,880
```

#### 3 Final Layer Norm
- After 30 layers of adding and multiplying numbers, the values in the "thought vector" might have drifted. Some might be huge (e.g., 500.0), some tiny (e.g., 0.001). The Final Layer Norm (RMSNorm) forces these numbers back into a standard range (e.g. variance of 1).
- It ensures the "thought" is clean and standardized before it tries to match it to a word.
```
576
```

#### 4 LM Head (Output Projection)
As we have ```tie_word_embeddings = True```
- It shares weights with embedding layer
- 0 additional parameters

#### 5 Final Total
```
┌────────────────────────────────────────────────────┐
│ Component                    │ Parameters          │
├────────────────────────────────────────────────────┤
│ Embedding Layer              │ 28,311,552          │
│ 30 Transformer Blocks        │ 106,202,880         │
│ Final Layer Norm             │ 576                 │
│ LM Head (tied)               │ 0                   │
├────────────────────────────────────────────────────┤
│ TOTAL                        │ 134,515,008         │
│                              │ ≈ 135M ✓            │
└────────────────────────────────────────────────────┘
```

### Things to know
1. Trained on ~41 Million tokens!
```
block_size = 1024    # Model sees 1024 tokens at a time
batch_size = 8       # 8 samples processed together
max_steps  = 5000    # 5000 weight updates

# Total tokens trained
= 5000 × 8 × 1024
= 40,960,000 tokens  # ~41 Million tokens!
```
2. ```hidden_size=576``` divisible by 9 heads , enables clean head dim =64
3. ```num layers = 30```, deep and narrow, good for small models
3. ```tie_embeddings=True``` saves 28M parameters, efficient reuse for small models
4. ```num_kv_heads=3``` GQA (9Q, 3KV), reduces memory and maintains quality.

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

## The Roles: Q, K, and V
What these matrices actually do in simple terms -

* Query (Q) = The Analyst asking a question. ("I am looking for dates," "I am looking for names," "I am looking for verbs.")

* Key (K) = The File Folder Label. ("This folder contains dates," "This folder contains names.")

* Value (V) = The Content. (The actual papers inside the folder.)

The attention mechanism is simply the Analyst (Q) matching their question to the right Label (K) to retrieve the Content (V).


## Known issues: 
1. Fix validation dataset : There should be a split of data from the file and validatin dataset should be different.
    - Current using a small set of same train set.
2. During validation : CPU/ GPU is not used efficiently
