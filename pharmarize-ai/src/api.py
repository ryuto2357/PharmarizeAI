"""
Pharmarize.ai REST API
Flask backend for Q&A chatbot - Phase 3 integration ready
"""

from flask import Flask, request, jsonify
import os
import logging
from qa_engine import PharmarizeQAEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global Q&A engine instance
qa_engine = None


def load_qa_engine():
    """Load Q&A engine on startup"""
    global qa_engine
    try:
        model_path = os.getenv("MODEL_PATH", "./models/pharmarize_qa_model")
        device = "cuda" if os.getenv("USE_GPU") == "true" else "cpu"
        qa_engine = PharmarizeQAEngine(model_path, device)
        logger.info("✓ Q&A Engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Q&A Engine: {e}")
        raise


@app.before_request
def startup():
    """Initialize engine before first request"""
    if qa_engine is None:
        load_qa_engine()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Pharmarize.ai Q&A API",
        "version": "1.0.0"
    }), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Main Q&A endpoint

    Expected JSON:
    {
        "question": "Apa manfaat pasak bumi?",
        "context": "Tumbuhan pasak bumi..."
    }

    Returns:
    {
        "answer": "eurycomanone",
        "confidence": 0.95,
        "question": "Apa manfaat pasak bumi?"
    }
    """
    try:
        # Parse request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        question = data.get("question")
        context = data.get("context")

        # Validate inputs
        if not question or not context:
            return jsonify({
                "error": "Missing required fields: 'question' and 'context'"
            }), 400

        if len(question) < 3:
            return jsonify({"error": "Question too short"}), 400

        if len(context) < 20:
            return jsonify({"error": "Context too short"}), 400

        # Get answer
        result = qa_engine.answer_question(question, context)

        # Format response
        response = {
            "question": question,
            "answer": result.get("answer"),
            "confidence": float(result.get("confidence", 0)),
            "success": True
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/batch-ask', methods=['POST'])
def batch_ask():
    """
    Batch Q&A endpoint

    Expected JSON:
    {
        "qa_pairs": [
            {
                "question": "Q1",
                "context": "Context1"
            },
            {
                "question": "Q2",
                "context": "Context2"
            }
        ]
    }
    """
    try:
        data = request.get_json()
        qa_pairs = data.get("qa_pairs", [])

        if not qa_pairs:
            return jsonify({"error": "No qa_pairs provided"}), 400

        questions = [pair.get("question") for pair in qa_pairs]
        contexts = [pair.get("context") for pair in qa_pairs]

        # Get answers
        results = qa_engine.batch_answer(questions, contexts)

        response = {
            "total": len(results),
            "results": [
                {
                    "question": q,
                    "answer": result.get("answer"),
                    "confidence": float(result.get("confidence", 0))
                }
                for q, result in zip(questions, results)
            ],
            "success": True
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in /batch-ask endpoint: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": "indobenchmark/indobert-base-p1",
        "task": "question_answering",
        "framework": "transformers",
        "input_format": "SQuAD",
        "max_sequence_length": 384,
        "device": "cpu"
    }), 200


@app.route('/version', methods=['GET'])
def version():
    """Get API version"""
    return jsonify({
        "version": "1.0.0",
        "service": "Pharmarize.ai Q&A API",
        "phase": "3_grand_final",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500


if __name__ == '__main__':
    # Load engine on startup
    load_qa_engine()

    # Run server
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting Pharmarize.ai API on port {port}")
    logger.info(f"Debug mode: {debug}")

    app.run(
        host='127.0.0.1',
        port=port,
        debug=debug,
        threaded=True
    )
