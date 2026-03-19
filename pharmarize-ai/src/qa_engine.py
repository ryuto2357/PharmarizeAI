"""
Pharmarize.ai Q&A Engine
Core inference module for question answering using fine-tuned IndoBERT
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PharmarizeQAEngine:
    """
    Q&A Engine using fine-tuned IndoBERT
    Handles context retrieval and answer extraction
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the Q&A engine

        Args:
            model_path: Path to fine-tuned model directory
            device: "cpu" or "cuda"
        """
        self.device = device
        self.model_path = model_path

        # Load tokenizer and model
        logger.info(f"Loading model from {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            self.model.to(device)
            self.model.eval()
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def answer_question(
        self,
        question: str,
        context: str,
        top_k: int = 1
    ) -> Dict:
        """
        Answer a question given a context

        Args:
            question: User's question in Indonesian
            context: Document context to search for answer
            top_k: Number of top answers to return

        Returns:
            Dictionary with answer, confidence, and span info
        """
        # Tokenize input
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=384,
            padding="max_length",
            truncation=True
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask
            )
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        # Extract answer span
        start_scores, start_idx = torch.topk(start_logits, top_k)
        end_scores, end_idx = torch.topk(end_logits, top_k)

        # Decode answer
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        answer_tokens = tokens[start_idx[0]:end_idx[0]+1]
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)

        # Calculate confidence score
        confidence = (
            float(start_scores[0]) + float(end_scores[0])
        ) / 2.0

        return {
            "answer": answer.strip(),
            "confidence": confidence,
            "start_idx": int(start_idx[0]),
            "end_idx": int(end_idx[0]),
            "context_used": context[:100] + "..." if len(context) > 100 else context
        }

    def batch_answer(
        self,
        questions: List[str],
        contexts: List[str]
    ) -> List[Dict]:
        """
        Answer multiple questions

        Args:
            questions: List of questions
            contexts: List of contexts (one per question)

        Returns:
            List of answer dictionaries
        """
        results = []
        for question, context in zip(questions, contexts):
            try:
                result = self.answer_question(question, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error answering '{question}': {e}")
                results.append({"error": str(e)})

        return results


# Example usage
if __name__ == "__main__":
    # This is a template - actual model path depends on training
    engine = PharmarizeQAEngine(
        model_path="./models/pharmarize_qa_model",
        device="cpu"
    )

    # Example Q&A
    context = "Tumbuhan pasak bumi mengandung eurycomanone yang bermanfaat untuk stamina."
    question = "Apa itu pasak bumi?"

    result = engine.answer_question(question, context)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
