# Pharmarize.ai - Step-by-Step Environment Setup

This guide walks you through setting up the development environment for the Pharmarize.ai project.

## Prerequisites

- Python 3.8+ (3.10+ recommended)
- 8-16GB RAM (for training)
- Git
- Internet connection (for first-time downloads)

## Step 1: Clone & Navigate

```bash
# Navigate to your project directory
cd /home/nahida/Binus/2026-03-19_PharmarizeAI/pharmarize-ai

# Verify directory structure
ls -la
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Linux/Mac)
source venv/bin/activate

# OR activate it (Windows)
venv\Scripts\activate

# Verify activation (you should see (venv) in your terminal)
```

## Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version - smaller, faster for most machines)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ML libraries
pip install transformers datasets pandas numpy scikit-learn

# Install Jupyter & notebooks
pip install jupyterlab notebook

# Install additional dependencies
pip install sentence-transformers python-dotenv flask requests PyYAML

# Verify all installed
pip list
```

## Step 4: Verify Installation

```bash
# Start Jupyter Lab
jupyter lab

# In browser, open: notebooks/01_verify_setup.ipynb
# Run all cells to verify setup
```

**Expected output**: All cells pass without errors

## Step 5: Prepare Data Directories

```bash
# Create subdirectories
mkdir -p data/raw_journals
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p results

# Verify
ls -R data/
```

## Step 6: Download Sample Data

### Option A: Manual (Recommended for first time)

1. Visit [Garba Rujukan Digital](https://garuda.kemdiktisaintek.go.id/)
2. Search: "tanaman obat endemic Indonesia" or "senyawa aktif"
3. Download PDFs
4. Save to `data/raw_journals/`

**Sample search terms**:
- "tanaman obat" (medicinal plants)
- "endemic Indonesia" (endemic to Indonesia)
- "senyawa aktif" (active compounds)
- "ekstraksi" (extraction)
- "pasak bumi" (specific plant)

### Option B: Automated (Python script - coming later)

```python
# Placeholder for data download script
# See notebooks/02_finetune_indobert.ipynb for context
```

## Step 7: Create Plant Dictionary

Create `data/plant_dictionary.json`:

```json
{
  "plants": [
    {
      "local_name": "Pasak Bumi",
      "scientific_name": "Eurycoma longifolia",
      "region": "Southeast Asia",
      "compounds": ["Eurycomanone", "Alkaloids"],
      "uses": ["Stamina", "Fertility"],
      "source_journal": "journal_001.pdf"
    },
    {
      "local_name": "Kunyit",
      "scientific_name": "Curcuma longa",
      "region": "Indonesia",
      "compounds": ["Curcumin"],
      "uses": ["Anti-inflammatory", "Digestive"],
      "source_journal": "journal_002.pdf"
    }
  ]
}
```

## Step 8: Test IndoBERT Loading

```bash
# Start Python
python

# In Python shell:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
print("IndoBERT loaded successfully!")
exit()
```

**Note**: First load will download (~500MB), subsequent loads use cache

## Step 9: Start Development

```bash
# Open Jupyter Lab
jupyter lab

# Create new notebooks in notebooks/ directory
# Or use provided notebooks as templates
```

## Troubleshooting

### PyTorch Installation Issues

If PyTorch fails to install:
```bash
# Try without CPU index URL
pip install torch torchvision torchaudio

# Or specify older version
pip install torch==2.0.0
```

### Memory Issues During Training

If you run out of RAM:
```python
# In notebook, reduce batch size
batch_size = 4  # Instead of 8

# Or reduce max_seq_length
max_seq_length = 256  # Instead of 384
```

### IndoBERT Won't Load

- **First time?** It downloads ~500MB - be patient
- **No internet?** Pre-download using: `python -m transformers.utils.hub_cli <model_name>`
- **Disk space?** Models cache to `~/.cache/huggingface/`

### Jupyter Lab Won't Start

```bash
# Try Jupyter Notebook instead
jupyter notebook

# Or check for port conflicts
jupyter lab --port 8889
```

## Environment Variables

Create `.env` file for configuration:

```bash
# .env
MODEL_NAME="indobenchmark/indobert-base-p1"
DATA_PATH="./data/processed"
BATCH_SIZE=8
LEARNING_RATE=2e-5
EPOCHS=3
```

Load in Python:
```python
from dotenv import load_dotenv
import os
load_dotenv()
model_name = os.getenv("MODEL_NAME")
```

## Next Steps

1. ✅ Environment setup complete
2. → Run `01_verify_setup.ipynb` to test
3. → Collect data from Garba Rujukan Digital
4. → Follow `02_finetune_indobert.ipynb` for training
5. → Test inference with `03_inference_demo.ipynb`

## Quick Command Reference

```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter lab

# Run Python script
python src/api.py

# Deactivate environment
deactivate

# Check installed packages
pip list

# Update package
pip install --upgrade package_name

# Remove package
pip uninstall package_name
```

## Team Collaboration

**Use Git for version control**:
```bash
git add .
git commit -m "Setup environment"
git push origin main
```

**Share requirements**:
```bash
# Generate requirements after installing new packages
pip freeze > requirements.txt
```

## Getting Help

1. Check console error messages carefully
2. Search error message on Google/Stack Overflow
3. Check HuggingFace forums
4. Ask team members
5. Review plan file: `/home/nahida/.claude/plans/enumerated-greeting-porcupine.md`

---

**Once complete, you're ready to start Phase 1: Data Collection!**
