# How to Train IndoBERT for Q&A - Complete Guide

Welcome! This guide teaches you the concepts BEFORE you run the notebook.

---

## 🎯 The Goal

We want to train a model that:
- **Reads** a question in Indonesian
- **Reads** a document/context text
- **Finds and extracts** the answer from that document

**Example:**
```
Question: "Apa kandungan pasak bumi?"
Context: "Tumbuhan pasak bumi mengandung eurycomanone yang bermanfaat untuk stamina."
Answer: "eurycomanone" ← Our model extracts this
```

This is called **Extractive Question Answering** - we're NOT generating new text, just extracting existing parts.

---

## 📚 Understanding the Training Process

### Step 1: Prepare Your Data

**Format: SQuAD (Stanford Question Answering Dataset)**

Your data must look like this:
```json
{
  "data": [
    {
      "title": "Pasak Bumi",
      "paragraphs": [
        {
          "context": "Tumbuhan pasak bumi mengandung eurycomanone...",
          "qas": [
            {
              "question": "Apa kandungan pasak bumi?",
              "id": "q1",
              "answers": [
                {
                  "text": "eurycomanone",
                  "answer_start": 36  ← Character position in context
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**The `answer_start` is critical**: It tells the model exactly WHERE in the context the answer is. This position must be counted **by character**, not by word.

### Step 2: Tokenization

The model doesn't understand text - it understands **tokens** (small pieces of text).

**Example:**
```
Question: "Apa kandungan pasak bumi?"
Tokens: ["Apa", "kan", "##dung", "an", "pasak", "bumi", "?"]
Token IDs: [103, 2478, 12889, 1097, 4534, 1234, 136]
```

Notice `##dung` - that's a subword token (the `##` means it's part of a larger word).

**Our job during tokenization:**
1. Convert user's question to tokens
2. Convert user's context to tokens
3. Find WHERE the answer tokens are in the sequence
4. Record the starting token index and ending token index

This is the tricky part! We have to convert character positions → token positions.

### Step 3: The Model

IndoBERT is a **Transformer model** with this architecture:

```
[Question] [Context containing answer]
    ↓
[Transformer layers - 12 layers of neural networks]
    ↓
[Start position logits] [End position logits]
    ↓
Where does answer start? Where does it end?
```

The model learns to output:
- **Start logits**: "The answer probably starts at token position 6"
- **End logits**: "The answer probably ends at token position 9"

Then we extract tokens 6-9 and convert back to text: "eurycomanone"

### Step 4: Training

**What happens during training:**

1. **Forward Pass**: Feed data through model
   ```
   Question + Context → Model → Start/End predictions
   ```

2. **Calculate Loss**: "How wrong was the model?"
   ```
   Loss = |Predicted_start - True_start| + |Predicted_end - True_end|
   ```

3. **Backward Pass**: Calculate gradients (which direction to move weights)

4. **Update Weights**: Adjust model parameters to reduce loss
   ```
   New_weights = Old_weights - learning_rate * gradients
   ```

5. **Repeat** for 3 epochs (3 times through entire dataset)

**Key hyperparameters:**

| Parameter | What it does | Our value |
|-----------|-------------|-----------|
| Learning Rate | How big a step to take when updating weights | 2e-5 |
| Batch Size | How many examples before updating | 8 |
| Epochs | How many times to see entire dataset | 3 |
| Max Seq Length | Maximum tokens to process | 384 |

---

## 🧠 Simplified Training Flow

```
1. Load Data
   ↓
2. Preprocess (tokenize + find answer positions)
   ↓
3. Load Pre-trained IndoBERT
   ↓
4. Set training parameters (learning rate, batch size, etc)
   ↓
5. TRAIN:
   For epoch 1, 2, 3:
     For each batch of 8 examples:
       - Forward pass
       - Calculate loss
       - Backward pass
       - Update weights
       - Print progress
   ↓
6. EVALUATE on validation set
   ↓
7. SAVE the fine-tuned model
   ↓
8. TEST on new questions
```

---

## 🎓 What is "Fine-Tuning"?

**Pre-training** (done by researchers at IndonesianBERT):
- Trained on billions of words of Indonesian text
- Learned general language understanding
- Took weeks on powerful GPUs

**Fine-tuning** (what we do):
- Start with pre-trained weights
- Add Q&A-specific head (2 small neural networks)
- Train only on Q&A data for 3 epochs
- Takes minutes on CPU
- Much cheaper and faster!

Why does this work? The model already knows how to read Indonesian. We're just teaching it where answers are located.

---

## 💡 Key Concepts Explained Simply

### 1. Loss Function

Loss = How wrong the model is.

```
If true answer is at positions [6, 9]:
Model predicts [5, 10]
Loss = |5-6| + |10-9| = 2 (pretty wrong!)

Model predicts [6, 9]
Loss = |6-6| + |9-9| = 0 (perfect!)
```

Goal: Minimize loss during training.

### 2. Validation Set

Data we DON'T train on, but evaluate the model with:
- Prevents overfitting (memorizing training data)
- Tells us real-world performance
- We save the best model based on validation loss

### 3. Epochs

One epoch = seeing every training example once.

```
Epoch 1: See all 70 examples, update weights
Epoch 2: See all 70 examples again, update weights more
Epoch 3: See all 70 examples again, update weights more
```

Why 3 epochs? Common rule is 3-5 for fine-tuning small Q&A datasets.

### 4. Batch Size = 8

Process 8 examples, calculate loss, update weights, repeat.

```
Example 1 ┐
Example 2 │
Example 3 │
Example 4 ├─ Calculate loss → Update weights
Example 5 │
Example 6 │
Example 7 │
Example 8 ┘

Example 9 ┐
...
```

Why 8? Larger batches = more stable but slower. 8 is sweet spot for 8-16GB RAM.

---

## 📊 Understanding Training Metrics

### Loss

The error value - we want this to GO DOWN.

```
Epoch 1: Loss = 3.45
Epoch 2: Loss = 2.10  ← Getting better!
Epoch 3: Loss = 1.67  ← Getting even better!
```

If loss goes UP, training is failing (maybe learning rate too high).

### Exact Match (EM)

Did we predict the EXACT answer?

```
Context: "Du Bois was born in 1868"
Question: "When was Du Bois born?"
True answer: "1868"

Model prediction: "1868" → EM = 1 (correct!)
Model prediction: "Du Bois was born in 1868" → EM = 0 (wrong, too long)
Model prediction: "1867" → EM = 0 (close but wrong)
```

### F1 Score

How much of our answer overlaps with true answer (word-level).

```
True answer: "eurycomanone compound"
Model predicts: "eurycomanone"
F1 = 0.67 (got 2 out of 3 words approximately)
```

---

## 🔧 Hyperparameters - What to Adjust

**If model is not learning well:**
- Lower learning rate: `2e-5 → 1e-5` (slower but more stable)
- Increase epochs: `3 → 5`
- Check if data is correct format

**If memory error:**
- Lower batch size: `8 → 4`
- Lower max_seq_length: `384 → 256`

**If overfitting (train loss↓ but eval loss↑):**
- Increase weight_decay: `0.01 → 0.1`
- Add more data
- Decrease epochs

---

## 📝 Your Q&A Data Checklist

Before running the notebook, make sure:

- [ ] 70% of data in train folder
- [ ] 30% of data in eval folder
- [ ] Each example has `question`, `context`, `answer`, `answer_start`
- [ ] `answer_start` is correct character position
- [ ] All answers ARE in the context (not impossible-to-find)
- [ ] Format matches SQuAD exactly
- [ ] Saved as `qa_dataset.json`

---

## 🚀 Running the Notebook

**Order matters!**

1. ✅ Run cell by cell, don't skip
2. ✅ Read each markdown section (explanations)
3. ✅ Check outputs at each step
4. ⏸ If something fails, read the error carefully
5. ✅ Once done, your model is ready at `models/pharmarize_qa_model/`

**Expected time:**
- With sample data: 2-5 minutes
- With 100+ real examples: 10-20 minutes on CPU

---

## ✅ After Training: What's Next?

1. **Check metrics**: Is validation loss low? Is EM score reasonable?

2. **Test inference notebook** (03_inference_demo.ipynb):
   - Ask your fine-tuned model new questions
   - See if it answers correctly

3. **Save your model**:
   - Already saved at `models/pharmarize_qa_model/`
   - Can be shared and deployed

4. **If results are bad**:
   - Get more training data
   - Check if data format is correct
   - Try different hyperparameters
   - See training tips section

5. **When ready**:
   - Use in Phase 3 API (`src/api.py`)
   - Deploy for the competition!

---

## 🆘 Common Problems & Solutions

### Problem: "answer not found in context"
**Solution**: Your `answer_start` position is wrong. Count characters carefully!

### Problem: Loss doesn't decrease
**Solution**: Learning rate too high. Try `1e-5` instead of `2e-5`

### Problem: Out of memory error
**Solution**: Decrease batch size to 4 or decrease max_seq_length to 256

### Problem: Model predicts same answer every time
**Solution**: Dataset too small or imbalanced. Need more diverse questions

### Problem: Very high loss but no convergence
**Solution**: Check if tokenizer matches model. Or check data format.

---

## 📚 Learning Resources

- [HuggingFace Documentation](https://huggingface.co/docs/transformers/)
- [SQuAD Task Explanation](https://rajpurkar.github.io/SQuAD-explorer/)
- [IndoBERT Paper](https://arxiv.org/abs/2009.05387)
- [Fine-tuning Guide](https://huggingface.co/course/chapter3/)

---

## TL;DR - Quick Summary

**Training in 3 sentences:**
1. You prepare Q&A data in SQuAD format with correct answer positions
2. You run the notebook which loops through data multiple times, updating model weights
3. After training, you get a fine-tuned model that can answer questions about your documents

**Key hyperparameters to remember:**
- Learning Rate: 2e-5 (how fast to learn)
- Batch Size: 8 (how many before update)
- Epochs: 3 (how many times to see data)
- Max Seq: 384 (token limit)

**Success metrics:**
- Validation loss should decrease
- EM score > 0.5 is good for small datasets
- Model should answer training examples reasonably

Now open the notebook and follow along! 🚀
