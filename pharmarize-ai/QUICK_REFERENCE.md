# 🚀 Quick Reference: Training IndoBERT

## Before You Run

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Check data format
ls data/qa_dataset.json  # Must exist!

# 3. Start Jupyter
jupyter lab

# 4. Open: notebooks/02_finetune_indobert.ipynb
```

---

## Data Format Check

Your JSON must have this structure:
```json
{
  "data": [
    {
      "title": "Article Name",
      "paragraphs": [
        {
          "context": "Long text containing the answer...",
          "qas": [
            {
              "question": "Your question?",
              "id": "unique_id",
              "answers": [
                {
                  "text": "answer",
                  "answer_start": 123
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

**Critical**: `answer_start` = character position where answer begins (not word position!)

---

## Hyperparameters Cheat Sheet

| Parameter | Value | Meaning |
|-----------|-------|---------|
| MODEL_NAME | indobenchmark/indobert-base-p1 | The model |
| BATCH_SIZE | 8 | Examples per update |
| LEARNING_RATE | 2e-5 | How fast to learn |
| EPOCHS | 3 | Times through data |
| MAX_SEQ_LENGTH | 384 | Token limit |

**Quick fixes:**
- Memory error? → Batch size 8→4, max_len 384→256
- Loss not decreasing? → Learning rate 2e-5→1e-5
- Overfitting? → Add more epochs: 3→5
- Need faster training? → Batch size 8→16 (if memory allows)

---

## Training Pipeline (What The Notebook Does)

```
1️⃣  Load Data
    └─ Read qa_dataset.json

2️⃣  Preprocess
    └─ Tokenize: Text → Token IDs
    └─ Find answer positions in tokens

3️⃣  Load Model
    └─ Download IndoBERT from HuggingFace
    └─ Add Q&A head

4️⃣  Configure Training
    └─ Set batch size, learning rate, etc.

5️⃣  TRAIN (3 epochs)
    ├─ Epoch 1: See all data, update weights
    ├─ Epoch 2: See all data, update weights more
    └─ Epoch 3: See all data, update weights more

6️⃣  EVALUATE
    └─ Test on validation data
    └─ Calculate loss, EM score

7️⃣  SAVE
    └─ Save to models/pharmarize_qa_model/

8️⃣  TEST
    └─ Try on sample questions
```

---

## Expected Output

### Good Training:
```
Epoch 1/3
Training loss: 2.34  ← Getting lower
Validation loss: 2.10

Epoch 2/3
Training loss: 1.45
Validation loss: 1.32

Epoch 3/3
Training loss: 0.98
Validation loss: 1.05
```

### Bad Training:
```
❌ Training loss: 3.45, 3.44, 3.43  ← Not decreasing (check data!)
❌ Training loss: 5.67  ← Exploding (learning rate too high!)
❌ CUDA out of memory  ← Batch size too large
```

---

## File Structure During Training

```
pharmarize-ai/
├── data/
│   └── qa_dataset.json ← YOUR DATA HERE
├── models/
│   ├── checkpoints/
│   │   ├── checkpoint-1/  ← Epoch 1 model
│   │   ├── checkpoint-2/  ← Epoch 2 model
│   │   └── checkpoint-3/  ← Epoch 3 model (best one)
│   └── pharmarize_qa_model/  ← FINAL MODEL HERE
├── results/
│   └── training_metrics.json ← Loss values, etc.
```

---

## Key Metrics to Watch

| Metric | What it means | Good value |
|--------|--------------|------------|
| training_loss | Model's error during training | < 1.0 |
| eval_loss | Model's error on unseen data | < 1.5 |
| EM (Exact Match) | % of perfect predictions | > 50% |
| F1 Score | Better metric for close answers | > 60% |

---

## Failure Troubleshooting

### "answer_start" error
```
Error: "answer not found in context"
Fix:
1. Go to your JSON
2. Find the question that failed
3. Count characters in context CAREFULLY
4. Position 0 = first character 'T' in "Tumbuhan..."
5. Recount and fix answer_start
```

### Training hangs or very slow
```
Problem: Running on CPU takes forever
Solution:
- This is normal! CPU training for 100+ examples = 20+ minutes
- If it's completely stuck: check STILL printing updates
- Ctrl+C and restart if truly stuck > 1 hour
```

### Out of Memory
```
Error: "CUDA out of memory" or system freezes
Fix:
1. Reduce batch size: BATCH_SIZE = 4 (or lower)
2. Reduce max length: MAX_SEQ_LENGTH = 256
3. Kill other programs
4. Restart Jupyter and try again
```

### Loss doesn't decrease
```
Problem: Training loss stays at 3.45, 3.44, 3.44...
Solution:
1. Check answer_start values (most common!)
2. Try lower learning rate: 1e-5
3. Verify JSON format is correct
4. Check tokenizer matches model
```

### Model predicts same answer always
```
Problem: Output always "eurycomanone"
Reason: Dataset too unbalanced or too small
Solution:
1. Add more training data (50+ examples minimum)
2. Mix different question types
3. Vary document lengths
```

---

## After Training: Next Steps

### ✅ Quick Validation
```python
# In notebook after training:
result = predict_answer(
    question="Your test question?",
    context="Some context here",
    model=inference_model,
    tokenizer=inference_tokenizer
)
print(result)  # Should show reasonable answer
```

### ✅ View Results File
```bash
cat results/training_metrics.json  # See all metrics
```

### ✅ For Phase 3 Integration
```python
# Your model is ready at:
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained(
    "models/pharmarize_qa_model"
)
# Now use with Flask API in src/api.py
```

---

## Commands You'll Use

```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter lab

# View metrics after training
cat results/training_metrics.json

# Check model size
du -sh models/pharmarize_qa_model/

# List checkpoints created
ls models/checkpoints/

# Stop Jupyter (in terminal)
Ctrl+C

# Deactivate environment
deactivate
```

---

## Timeline

| Phase | Time | What |
|-------|------|------|
| Data prep | 5 min | Load data, split into train/val |
| Preprocessing | 2 min | Tokenize and find answer positions |
| Model loading | 1 min | Download IndoBERT |
| Training | 2-20 min | Main training loop |
| Evaluation | 1 min | Calculate metrics |
| Save | 1 min | Save fine-tuned model |
| **TOTAL** | **10-30 min** | **On CPU** |

---

## Remember These!

✅ **Must have** before running:
- `data/qa_dataset.json` with correct format
- Answer positions verified
- Virtual environment activated

✅ **During training**, watch for:
- Training loss going DOWN
- No memory errors
- Validation loss reasonable

✅ **After training**:
- Check metrics.json file
- Test on sample questions
- Save the model (already done!)

✅ **For Phase 3**:
- Your model is in `models/pharmarize_qa_model/`
- Ready to load and use with API
- No retraining needed during competition!

---

## 🎯 Success Checklist

- [ ] Data in correct SQuAD JSON format
- [ ] Answer positions are character positions, not word positions
- [ ] Answers are actually in the context
- [ ] Training loss decreased over 3 epochs
- [ ] Model saved at `models/pharmarize_qa_model/`
- [ ] Can run test questions and get answers back
- [ ] Metrics saved to `results/training_metrics.json`

**If all checked: You're ready for Phase 3!** 🚀
