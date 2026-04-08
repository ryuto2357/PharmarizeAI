#!/usr/bin/env python3
"""
IndoBERT Fine-tuning Script for Pharmarize.ai
Train a Question Answering model on Indonesian medicinal plant data.

This script shows real-time training progress.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from transformers.trainer_callback import TrainerCallback
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "data" / "qa_dataset.json"
OUTPUT_DIR = SCRIPT_DIR / "models" / "pharmarize_qa_model"
CHECKPOINT_DIR = SCRIPT_DIR / "models" / "checkpoints"


# ============================================
# PROGRESS CALLBACK
# ============================================
class ProgressCallback(TrainerCallback):
    """Custom callback to show training progress."""
    
    def __init__(self):
        self.current_epoch = 0
        self.total_steps = 0
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.total_steps = state.max_steps
        print("\n" + "="*60)
        print("🚀 TRAINING STARTED")
        print("="*60)
        print(f"   Total epochs: {args.num_train_epochs}")
        print(f"   Total steps: {state.max_steps}")
        print(f"   Batch size: {args.per_device_train_batch_size}")
        print(f"   Learning rate: {args.learning_rate}")
        print("="*60 + "\n")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch += 1
        print(f"\n📚 EPOCH {self.current_epoch}/{int(args.num_train_epochs)}")
        print("-"*40)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            progress = (step / self.total_steps) * 100
            
            # Create progress bar
            bar_length = 30
            filled = int(bar_length * step / self.total_steps)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Get metrics
            loss = logs.get("loss", logs.get("eval_loss", 0))
            
            # Calculate ETA
            if self.start_time and step > 0:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                eta_seconds = (elapsed / step) * (self.total_steps - step)
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = f"{eta_min}m {eta_sec}s"
            else:
                eta_str = "calculating..."
            
            print(f"   [{bar}] {progress:5.1f}% | Step {step}/{self.total_steps} | Loss: {loss:.4f} | ETA: {eta_str}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"   ✓ Epoch {self.current_epoch} complete!")
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n   📊 Validation Results:")
            print(f"      Loss: {metrics.get('eval_loss', 0):.4f}")
            
    def on_train_end(self, args, state, control, **kwargs):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"   Total time: {elapsed_min}m {elapsed_sec}s")
        print(f"   Final loss: {state.log_history[-1].get('loss', 'N/A')}")
        print("="*60 + "\n")


# ============================================
# DATASET CLASS
# ============================================
class QADataset(Dataset):
    """Dataset for Question Answering task."""
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


# ============================================
# DATA PREPARATION
# ============================================
def load_and_prepare_data(data_path, tokenizer):
    """Load QA dataset and prepare for training."""
    
    print("📂 Loading dataset...")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Flatten the data
    examples = []
    for doc in raw_data["data"]:
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answer_text": qa["answers"][0]["text"],
                    "answer_start": qa["answers"][0]["answer_start"]
                })
    
    print(f"   Total examples: {len(examples)}")
    
    # Split into train/validation (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    split_idx = int(len(examples) * 0.8)
    
    train_examples = [examples[i] for i in indices[:split_idx]]
    val_examples = [examples[i] for i in indices[split_idx:]]
    
    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")
    
    # Tokenize
    print("\n🔤 Tokenizing data...")
    train_encodings = tokenize_examples(train_examples, tokenizer)
    val_encodings = tokenize_examples(val_examples, tokenizer)
    
    return QADataset(train_encodings), QADataset(val_encodings)


def tokenize_examples(examples, tokenizer):
    """Tokenize examples and find answer positions in tokens."""
    
    questions = [ex["question"] for ex in examples]
    contexts = [ex["context"] for ex in examples]
    
    # Tokenize
    encodings = tokenizer(
        questions,
        contexts,
        max_length=MAX_SEQ_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=False,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Find start and end positions
    start_positions = []
    end_positions = []
    
    for i, ex in enumerate(examples):
        answer_start_char = ex["answer_start"]
        answer_end_char = answer_start_char + len(ex["answer_text"])
        
        # Get offset mapping for this example
        offsets = encodings["offset_mapping"][i]
        
        # Find token positions
        start_token = None
        end_token = None
        
        for idx, (start, end) in enumerate(offsets):
            if start <= answer_start_char < end:
                start_token = idx
            if start < answer_end_char <= end:
                end_token = idx
                break
        
        # If answer not found in tokens, use CLS token (position 0)
        if start_token is None:
            start_token = 0
        if end_token is None:
            end_token = 0
            
        start_positions.append(start_token)
        end_positions.append(end_token)
    
    # Remove offset_mapping (not needed for training)
    encodings.pop("offset_mapping")
    
    # Add answer positions
    encodings["start_positions"] = start_positions
    encodings["end_positions"] = end_positions
    
    return encodings


# ============================================
# MAIN TRAINING FUNCTION
# ============================================
def train():
    """Main training function."""
    
    print("\n" + "="*60)
    print("🌿 PHARMARIZE.AI - IndoBERT Fine-tuning")
    print("="*60)
    print(f"   Model: {MODEL_NAME}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60 + "\n")
    
    # Load tokenizer and model
    print("📥 Loading IndoBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with num_labels=2 for QA (start and end positions)
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model loaded! Parameters: {param_count:,}")
    
    # Prepare data
    train_dataset, val_dataset = load_and_prepare_data(DATA_PATH, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard
        disable_tqdm=True,  # We use our own progress
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[ProgressCallback()],
    )
    
    # Train!
    trainer.train()
    
    # Save final model
    print("\n💾 Saving model...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"   Model saved to: {OUTPUT_DIR}")
    
    # Save training info
    info = {
        "model_name": MODEL_NAME,
        "trained_on": datetime.now().isoformat(),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "final_loss": trainer.state.log_history[-1].get("loss", None),
    }
    
    with open(OUTPUT_DIR / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n🎉 Training complete! Your model is ready.")
    print(f"   Location: {OUTPUT_DIR}")
    
    return trainer


if __name__ == "__main__":
    train()
