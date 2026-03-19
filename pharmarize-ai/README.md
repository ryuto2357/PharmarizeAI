# Pharmarize.ai - Q&A Chatbot for Indonesian Medicinal Plant Research

An offline, privacy-first Q&A chatbot powered by IndoBERT for answering questions about endemic Indonesian medicinal plants research.

## Project Overview

**Competition**: Find IT! 2026 Hackathon - Track B (Privacy Brain)
**Task**: Build a local NLP model for Question Answering on Indonesian plant research data
**Deadline**: April 12, 2026 (Phase 2) → May 14-15, 2026 (Phase 3 Finals)

## Quick Start

### 1. Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup
```bash
jupyter notebook notebooks/01_verify_setup.ipynb
# Run all cells to verify installation
```

### 3. Data Collection
- Download journals from [Garba Rujukan Digital](https://garuda.kemdiktisaintek.go.id/)
- Save PDFs to `data/raw_journals/`
- Search keywords: "tanaman obat", "endemic Indonesia", "senyawa aktif"

### 4. Training
```bash
jupyter notebook notebooks/02_finetune_indobert.ipynb
# Follow notebook for fine-tuning on Q&A task
```

### 5. Inference & API
```bash
# Test inference
jupyter notebook notebooks/03_inference_demo.ipynb

# Run Flask API (Phase 3)
python src/api.py
```

## Directory Structure

```
pharmarize-ai/
├── data/
│   ├── raw_journals/           # Original PDF files
│   ├── processed/              # Cleaned, tokenized texts
│   ├── plant_dictionary.json   # Local plant names mapping
│   └── qa_dataset.json         # Training Q&A data (SQuAD format)
├── models/
│   ├── checkpoints/            # Training checkpoints
│   └── pharmarize_qa_model/    # Final fine-tuned model
├── notebooks/
│   ├── 01_verify_setup.ipynb
│   ├── 02_finetune_indobert.ipynb
│   └── 03_inference_demo.ipynb
├── src/
│   ├── qa_engine.py            # Core Q&A inference engine
│   ├── api.py                  # Flask REST API
│   └── utils.py                # Helper functions
├── results/
│   ├── metrics.json            # Training metrics
│   └── training_logs/          # Detailed logs
├── requirements.txt
├── README.md
└── SETUP.md
```

## Technology Stack

- **Framework**: HuggingFace Transformers
- **Base Model**: IndoBERT (indobenchmark/indobert-base-p1)
- **ML Framework**: PyTorch (CPU)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **API**: Flask
- **Notebooks**: Jupyter Lab

## Model & Task

**Model**: IndoBERT Base (110M parameters)
- Pre-trained on Indonesian Wikipedia + Common Crawl
- Lightweight - runs on mid-range machines (8-16GB RAM)
- Optimized for Indonesian language understanding

**Task**: Question Answering (SQuAD-style)
- Extract answers from context given questions
- Metrics: Exact Match (EM) score, F1 score

## Key Features

✅ **Privacy-First**: All processing runs locally, no external APIs
✅ **Indonesian-Native**: IndoBERT understands local language
✅ **Lightweight**: Works on consumer hardware (no GPU needed)
✅ **Adaptable**: Modular design for Phase 3 integration
✅ **Documented**: Clear notebooks for each training stage

## Team Roles (4-5 people)

1. **Data Lead**: Collect journals, build plant dictionary
2. **ML Engineers** (x1-2): Fine-tuning, optimization
3. **Backend Engineer**: API, inference pipeline
4. **Product/Demo**: Testing, documentation

## Important Notes for Phase 3

- **No retraining during event**: Train model completely before May 14
- **Fast loading**: Model must load in <5 seconds
- **Flexible API**: Design to handle "Dynamic Injection" surprise variable
- **Standalone inference**: Can run on any machine without cloud dependency
- **Documented input/output**: Clear specifications for integration

## Timeline

| When | What |
|------|------|
| March 19-26 | Data collection, environment setup |
| March 26-Apr 2 | Initial fine-tuning, baseline results |
| April 2-12 | Optimization, API ready, full system testing |
| April 12 | Phase 2 Submission Deadline |
| May 14-15 | Phase 3 Grand Final (24-hour event) |

## References

- [IndoBERT Paper](https://arxiv.org/abs/2009.05387)
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers/)
- [SQuAD Task Format](https://rajpurkar.github.io/SQuAD-explorer/)
- [Garba Rujukan Digital](https://garuda.kemdiktisaintek.go.id/)

## Resources

- Plan: `/home/nahida/.claude/plans/enumerated-greeting-porcupine.md`
- Setup Guide: `SETUP.md`
- Architecture: `ARCHITECTURE.md` (TBD)

## Contact

See competition guidelines for team contact information.

---

**Made with ❤️ for Pharmarize.ai - Protecting Indonesian Research Data**
