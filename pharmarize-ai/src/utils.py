"""
Pharmarize.ai Utility Functions
Helper functions for data processing, evaluation, and utilities
"""

import json
import os
from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return {}


def save_json(data: Dict, filepath: str) -> bool:
    """Save dictionary to JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        return False


def load_plant_dictionary(dict_path: str) -> Dict[str, Dict]:
    """
    Load plant dictionary

    Returns:
        Dictionary mapping plant names to their properties
    """
    data = load_json(dict_path)
    plants_dict = {}

    for plant in data.get("plants", []):
        local_name = plant.get("local_name", "")
        scientific_name = plant.get("scientific_name", "")

        key = local_name.lower()
        plants_dict[key] = {
            "local_name": local_name,
            "scientific_name": scientific_name,
            "region": plant.get("region", ""),
            "compounds": plant.get("compounds", []),
            "uses": plant.get("uses", []),
            "source": plant.get("source_journal", "")
        }

    logger.info(f"Loaded {len(plants_dict)} plants from dictionary")
    return plants_dict


def clean_indonesian_text(text: str) -> str:
    """
    Clean Indonesian text for processing

    Args:
        text: Raw text string

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep Indonesian letters
    # ñ, á, é, í, ó, ú are handled by utf-8

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_qa_from_text(text: str, plant_name: str = None) -> List[Dict]:
    """
    Generate Q&A pairs from text (template-based)

    Args:
        text: Source text
        plant_name: Name of medicinal plant (optional)

    Returns:
        List of Q&A pairs in SQuAD format
    """
    qa_pairs = []

    # Template-based question generation
    templates = [
        ("Apa saja manfaat {plant}?", text),
        ("Apa kandungan aktif dalam {plant}?", text),
        ("Dari mana asal {plant}?", text),
        ("Bagaimana cara menggunakan {plant}?", text),
        ("Apa nama ilmiah {plant}?", text),
    ]

    # Generate questions if plant name provided
    if plant_name:
        for question_template, context in templates:
            question = question_template.format(plant=plant_name)

            qa_pairs.append({
                "question": question,
                "context": context,
                "answers": [{
                    "text": plant_name,  # Simplified - should be updated manually
                    "answer_start": context.find(plant_name)
                }]
            })

    return qa_pairs


def load_qa_dataset(dataset_path: str) -> Dict:
    """
    Load Q&A dataset in SQuAD format

    Args:
        dataset_path: Path to JSON dataset

    Returns:
        Dataset dictionary
    """
    dataset = load_json(dataset_path)
    logger.info(f"Loaded Q&A dataset with {len(dataset.get('data', []))} articles")
    return dataset


def save_qa_dataset(data: Dict, output_path: str) -> bool:
    """Save Q&A dataset"""
    return save_json(data, output_path)


def create_squad_format_qa(
    question: str,
    context: str,
    answer: str,
    answer_start: int
) -> Dict:
    """
    Create Q&A pair in SQuAD format

    Args:
        question: Question text
        context: Context/document text
        answer: Answer text
        answer_start: Character position where answer starts

    Returns:
        Q&A pair dict in SQuAD format
    """
    return {
        "question": question,
        "id": f"q_{hash(question) % 10000}",
        "answers": [{
            "text": answer,
            "answer_start": answer_start
        }],
        "is_impossible": False
    }


def calculate_metrics(predictions: List[Dict], references: List[Dict]) -> Dict:
    """
    Calculate EM (Exact Match) and F1 scores

    Args:
        predictions: Model predictions
        references: Ground truth references

    Returns:
        Dictionary with metrics
    """
    em_count = 0
    f1_scores = []

    for pred, ref in zip(predictions, references):
        pred_text = pred.get("answer", "").lower().strip()
        ref_text = ref.get("answer", "").lower().strip()

        # Exact Match
        if pred_text == ref_text:
            em_count += 1

        # F1 Score (token-level)
        pred_tokens = set(pred_text.split())
        ref_tokens = set(ref_text.split())

        if len(pred_tokens) + len(ref_tokens) == 0:
            f1 = 1.0 if pred_text == ref_text else 0.0
        else:
            common = len(pred_tokens & ref_tokens)
            f1 = 2 * common / (len(pred_tokens) + len(ref_tokens))

        f1_scores.append(f1)

    total = len(predictions)

    return {
        "exact_match": em_count / total if total > 0 else 0,
        "f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        "total_samples": total
    }


def debug_model_output(
    question: str,
    context: str,
    start_logits,
    end_logits,
    tokens: List[str]
) -> Dict:
    """
    Debug model output for analysis

    Args:
        question: Input question
        context: Input context
        start_logits: Model start position logits
        end_logits: Model end position logits
        tokens: Tokenized text

    Returns:
        Debug information dictionary
    """
    import torch

    return {
        "question": question,
        "context": context[:100],  # First 100 chars
        "num_tokens": len(tokens),
        "max_start_logit": float(torch.max(start_logits)),
        "max_end_logit": float(torch.max(end_logits)),
        "top_tokens": tokens[:10]
    }


def log_metrics(metrics: Dict, step: int = None):
    """Pretty print metrics"""
    prefix = f"Step {step}: " if step else ""
    print(f"{prefix}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


# Example usage
if __name__ == "__main__":
    # Test utility functions
    test_text = "Tumbuhan pasak bumi  mengandung   eurycomanone"
    cleaned = clean_indonesian_text(test_text)
    print(f"Cleaned text: {cleaned}")

    # Test Q&A creation
    qa = create_squad_format_qa(
        question="Apa kandungan pasak bumi?",
        context=cleaned,
        answer="eurycomanone",
        answer_start=cleaned.find("eurycomanone")
    )
    print(f"Q&A pair: {qa}")
