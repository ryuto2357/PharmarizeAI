#!/usr/bin/env python3
"""
Interactive Q&A Testing for Pharmarize.ai
Test your trained model with custom questions!
"""

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import json
from pathlib import Path

# Load model
print("📥 Loading Pharmarize.ai model...")
model_path = Path(__file__).parent.parent / "models" / "pharmarize_qa_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"✅ Model loaded on {device.upper()}!\n")

# Load sample contexts from our data
data_path = Path(__file__).parent.parent / "data" / "qa_dataset.json"
with open(data_path, 'r') as f:
    qa_data = json.load(f)

# Get some sample contexts
sample_contexts = []
for doc in qa_data["data"][:10]:
    for para in doc["paragraphs"][:2]:
        if len(para["context"]) > 200:
            sample_contexts.append({
                "title": doc["title"],
                "context": para["context"][:1000]
            })

def answer_question(question, context):
    """Get answer from model."""
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, max_length=384)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    
    # Make sure end >= start
    if end_idx < start_idx:
        end_idx = start_idx
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
    
    # Calculate confidence
    start_prob = torch.softmax(outputs.start_logits, dim=1)[0][start_idx].item()
    end_prob = torch.softmax(outputs.end_logits, dim=1)[0][end_idx].item()
    confidence = (start_prob + end_prob) / 2
    
    return answer.strip(), confidence

def main():
    print("="*60)
    print("🌿 PHARMARIZE.AI - Interactive Q&A Testing")
    print("="*60)
    print()
    print("Commands:")
    print("  [number] - Select a sample context (1-10)")
    print("  [c]      - Enter custom context")
    print("  [q]      - Quit")
    print()
    
    current_context = None
    current_title = None
    
    while True:
        # Show available contexts
        if current_context is None:
            print("-" * 60)
            print("📚 Available sample contexts:")
            for i, ctx in enumerate(sample_contexts[:10], 1):
                preview = ctx["context"][:80].replace("\n", " ")
                print(f"  [{i}] {ctx['title']}: {preview}...")
            print()
            
            choice = input("Select context [1-10] or [c]ustom or [q]uit: ").strip().lower()
            
            if choice == 'q':
                print("\n👋 Goodbye!")
                break
            elif choice == 'c':
                print("\nPaste your context (press Enter twice when done):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                current_context = " ".join(lines)
                current_title = "Custom"
                print(f"\n✅ Custom context loaded ({len(current_context)} chars)")
            elif choice.isdigit() and 1 <= int(choice) <= 10:
                idx = int(choice) - 1
                current_context = sample_contexts[idx]["context"]
                current_title = sample_contexts[idx]["title"]
                print(f"\n✅ Loaded: {current_title}")
            else:
                print("❌ Invalid choice")
                continue
        
        # Show current context
        print()
        print("-" * 60)
        print(f"📄 Current context: {current_title}")
        print("-" * 60)
        print(current_context[:500] + ("..." if len(current_context) > 500 else ""))
        print("-" * 60)
        print()
        
        # Ask questions
        while True:
            question = input("❓ Your question (or [b]ack, [q]uit): ").strip()
            
            if question.lower() == 'q':
                print("\n👋 Goodbye!")
                return
            elif question.lower() == 'b':
                current_context = None
                current_title = None
                print()
                break
            elif question == "":
                continue
            
            # Get answer
            answer, confidence = answer_question(question, current_context)
            
            # Display result
            print()
            print(f"💡 Answer: {answer}")
            print(f"📊 Confidence: {confidence:.1%}")
            
            # Confidence indicator
            if confidence > 0.7:
                print("   ✅ High confidence")
            elif confidence > 0.4:
                print("   ⚠️  Medium confidence")
            else:
                print("   ❓ Low confidence - answer may be inaccurate")
            print()

if __name__ == "__main__":
    main()
