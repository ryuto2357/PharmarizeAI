# Visual Guide: How IndoBERT Q&A Training Works

## Complete Training Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    PHARMARIZE.AI TRAINING                      │
└────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │  YOUR DATA  │
                         │ (JSON file) │
                         └──────┬──────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  SPLIT INTO TRAIN/VAL  │
                    │  70% training          │
                    │  30% validation        │
                    └────────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        ┌─────────────────┐          ┌──────────────────┐
        │  TRAIN DATA     │          │   VAL DATA       │
        │  (70 examples)  │          │  (30 examples)   │
        └────────┬────────┘          └──────────────────┘
                │
                ▼
        ┌─────────────────────────┐
        │  TOKENIZATION           │
        │  Text → Token IDs       │
        │  Find answer positions  │
        └────────────┬────────────┘
                │
                ▼
        ┌─────────────────────────┐
        │  LOAD INDOBERT MODEL    │
        │  (Download if needed)   │
        │  110M parameters        │
        └────────────┬────────────┘
                │
                ▼
        ┌─────────────────────────┐
        │  SET TRAINING PARAMS    │
        │  • learning_rate=2e-5   │
        │  • batch_size=8         │
        │  • epochs=3             │
        └────────────┬────────────┘
                │
                ▼
        ╔═════════════════════════╗
        ║   TRAIN FOR 3 EPOCHS    ║
        ╚════════════┬════════════╝
                     │
    ┌────────────────┴────────────────┐
    │                                 │
    ▼                                 ▼
(Epoch 1)                        (Epoch 1)
See all data            →         Update weights
    │                                 │
    ▼                                 ▼
(Epoch 2)                        (Epoch 2)
See all data            →         Update weights
    │                                 │
    ▼                                 ▼
(Epoch 3)                        (Epoch 3)
See all data            →         Update weights
    │                                 │
    └────────────────┬────────────────┘
                     ▼
            ┌──────────────────────┐
            │  VALIDATE MODEL      │
            │  Test on eval data   │
            │  Calculate metrics   │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  SAVE BEST MODEL     │
            │  Best checkpoint → final model
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  TEST ON NEW DATA    │
            │  Try sample Q&A      │
            └──────────┬───────────┘
                       │
                       ▼
        ✅ READY FOR PHASE 3!
```

---

## How the Model Works (One Example)

```
INPUT:
  Question:  "Apa kandungan pasak bumi?"
  Context:   "Tumbuhan pasak bumi mengandung eurycomanone..."

                        ▼▼▼ TOKENIZATION ▼▼▼

Tokens:  [CLS] apa kandungan pasak bumi ? [SEP] tumbuhan pasak bumi mengandung eurycomanone ...
IDs:     101  1234 2345  3456  4567  5678  102  6789  4567  3456  2345  6789 7890 ...
Pos:     0    1    2     3     4     5    6     7     8     9     10    11   12 ...


                 ▼▼▼ INDOBERT PROCESSES THIS ▼▼▼

12 transformer layers ↓↓↓

Each layer:"What's the context? Where might the answer be?"


                ▼▼▼ OUTPUT: START & END POSITIONS ▼▼▼

Model thinks:
  Start position = 11 (where "eurycomanone" starts)
  End position = 12 (where "eurycomanone" ends)

                ▼▼▼ EXTRACT ANSWER ▼▼▼

Tokens[11:13] = "eurycomanone"
Convert back to text = "eurycomanone"

                ▼▼▼ RESULT ▼▼▼

✅ ANSWER: "eurycomanone"
```

---

## One Training Step In Detail

```
┌──────────────────────────────────────────┐
│         SINGLE TRAINING STEP             │
│      (Happens 10+ times per epoch)       │
└──────────────────────────────────────────┘


BATCH: 8 Q&A examples together}
    │
    ├─ Example 1: Question + Context
    ├─ Example 2: Question + Context
    ├─ Example 3: Question + Context
    ├─ Example 4: Question + Context
    ├─ Example 5: Question + Context
    ├─ Example 6: Question + Context
    ├─ Example 7: Question + Context
    └─ Example 8: Question + Context
        │
        ▼
    [FORWARD PASS]
    All 8 → through model → 8 predictions
        │
        ▼
    [CALCULATE LOSS]
    For each example:
      loss = |predicted_start - true_start| + |predicted_end - true_end|

    Average across 8 examples = batch loss
        │
        ▼
    [BACKWARD PASS]
    Calculate gradients (which way to move weights)
        │
        ▼
    [UPDATE WEIGHTS]
    New_weights = Old_weights - learning_rate * gradients
        │
        ▼
    [REPEAT]
    Next batch of 8 examples...
```

---

## Training Curve (What You Want To See)

```
LOSS (Error)
  │
3 │  ●                                Good! Loss decreasing
  │    ●●
2 │      ●●
  │        ●●●
1 │           ●●●●
  │              ●●●●●●
0 │
  └─────────────────────────────── EPOCHS
    0   1   2   3   4   5   6   ...

Each ● = end of one batch (loss measurement)
You want: ↓↓↓ (downward trend)


❌ PROBLEMS TO WATCH FOR:

   Exploding Loss:              Not Converging:
   (Learning rate too high)     (Learning rate too low)

   3 │  ●                       3 │  ●●●●●●●●●●●
     │    ●●●●                   │  ●●●●●●●●●●●
   2 │        ●●●●●●             │  ●●●●●●●●●●●
     │             ●●●●●●●        │
   1 │                   ●●●●●   0 │
     │
   0 │
```

---

## Data Processing Pipeline

```
YOUR JSON FILE
    │
    ▼
┌─────────────────────────────────────┐
│  LOAD RAW DATA                      │
│  {data: [{title, paragraphs: ...}]} │
└─────────────┬───────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  EXTRACT Q&A PAIRS                   │
│  • question                          │
│  • context                           │
│  • answer_text                       │
│  • answer_start (character position) │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  TOKENIZE                            │
│  • Question → [101, 234, 345, ...]   │
│  • Context → [456, 567, 678, ...]    │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  FIND ANSWER TOKENS                  │
│  answer_start (chars) → token index  │
│  Example: "eurycomanone" at char 36  │
│           → starts at token 11       │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  CREATE TRAINING TENSORS             │
│  • input_ids                         │
│  • attention_mask                    │
│  • start_positions                   │
│  • end_positions                     │
└─────────────┬────────────────────────┘
              │
              ▼
        ✅ READY TO TRAIN!
```

---

## Model Architecture (Simplified)

```
INPUT: [CLS] + QUESTION + [SEP] + CONTEXT + [PAD]...
        │                                         │
        └─────────────────┬──────────────────────┘
                          │
             ┌────────────▼────────────┐
             │  EMBEDDING LAYER       │
             │  Text → Vectors        │
             │  (768 dimensions)      │
             └────────────┬────────────┘
                          │
    ┌─────────────────────┴─────────────────────┐
    │                                           │
    │  TRANSFORMER LAYERS (12 times)            │
    │  ┌──────────────┐  ┌──────────────┐       │
    │  │ Layer 1:     │→ │ Layer 2:     │→ ...  │
    │  │ Self-       │  │ Self-       │       │
    │  │ Attention   │  │ Attention   │       │
    │  └──────────────┘  └──────────────┘       │
    │  (Each layer learns more complex patterns)
    │                                           │
    └─────────────────────┬─────────────────────┘
                          │
                ┌─────────▼──────────┐
                │  OUTPUT LAYER      │
                │  768 → 2 classes   │
                │ (START or END)     │
                └────────────┬───────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
        [START LOGITS]                [END LOGITS]
        [3.2, -1.5, 4.1, ...]        [1.2, 3.8, 2.1, ...]
        Position 2 = 4.1 (highest)    Position 1 = 3.8 (highest)
        └─►Answer starts at token 2   └─►Answer ends at token 1
```

---

## Checkpoint System (Save Progress)

```
TRAINING PROGRESS:

Epoch 1:  Training...  Checkpoint-1 saved
          Val Loss: 2.10 ← Best so far!
              │
Epoch 2:  Training...  Checkpoint-2 saved
          Val Loss: 1.35 ← Even better!
              │
Epoch 3:  Training...  Checkpoint-3 saved
          Val Loss: 1.67 ← Slightly worse
              │
         ┌────┘
         ▼
    BEST MODEL = Checkpoint-2
    (Lowest validation loss)
         │
         ▼
    SAVE AS: models/pharmarize_qa_model/
    (This is what you use for Phase 3!)
```

---

## Batch Processing Visualization

```
Process 8 examples at once (with batch_size=8):

Q1 + C1 ─┐
Q2 + C2 ─┤
Q3 + C3 ─┤
Q4 + C4 ├─ Tokenize → Pass through model → Calculate loss
Q5 + C5 ├─ (All 8 together, faster than one by one)
Q6 + C6 ─┤
Q7 + C7 ─┤
Q8 + C8 ─┘

Loss_batch = (Loss_1 + Loss_2 + ... + Loss_8) / 8

Update weights once ← Efficient!

Then get next 8 examples...
```

---

## Memory Usage Over Time

```
MEMORY
  │
1 │  ┌────────────── Model (stays constant)
  │  │
  │  │  ┌───────────── Batch data (comes and goes)
  │  │  │
0 ├──┼──┼─────────────
  │  │  │
  └─────────────────────────── TIME
    Batch 1  Batch 2  Batch 3

With batch_size=8 and max_seq_length=384:
≈ 2GB per batch (varies)
Model = 0.4GB
Total free ≈ 2-3GB needed

That's why we need 8GB+ RAM!
```

---

## Metrics Explained (Side by Side)

```
METRIC 1: LOSS
┌────────────────────────────────────┐
│ Lower is better                    │
│ Measures: How wrong was model?     │
│ Range: 0 to ∞                      │
│ Good: < 1.5 after training         │
└────────────────────────────────────┘


METRIC 2: EXACT MATCH (EM)
┌────────────────────────────────────┐
│ Higher is better                   │
│ Measures: % perfect predictions    │
│ Range: 0% to 100%                  │
│ Good: > 50% for small dataset      │
│ Example:                           │
│  True: "eurycomanone"              │
│  Pred: "eurycomanone" ✓ EM=1       │
│  Pred: "euryco" ✗ EM=0 (not exact) │
└────────────────────────────────────┘


METRIC 3: F1 SCORE
┌────────────────────────────────────┐
│ Higher is better                   │
│ Measures: Word-level overlap       │
│ Range: 0 to 1                      │
│ Good: > 0.6 for small dataset      │
│ Example:                           │
│  True: "active compound"           │
│  Pred: "active" → F1=0.67 (ok)     │
│  Pred: "compound" → F1=0.67 (ok)   │
└────────────────────────────────────┘
```

---

Would you like me to explain any specific part in more detail? 🎓
