# 🩺 Med Vision Agent

**End-to-end Medical AI platform for Chest X-Ray analysis combining Deep Learning, Grad-CAM explainability, and RAG-powered clinical reporting.**

---

## Overview

Med Vision Agent is a production-grade medical AI system that takes a chest X-ray image and returns:
1. A **diagnostic classification** (Normal / Pneumonia) with a confidence score
2. A **Grad-CAM heatmap** highlighting regions the model focused on
3. An **AI-generated clinical report** synthesising the vision model's findings with evidence from clinical practice guidelines (via RAG)
4. An **interactive Q&A** interface for follow-up clinical questions

All served through a modern **Streamlit** web application.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                         │
│        Upload X-Ray → Diagnose → Report → Q&A           │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
   ┌───────────▼──────────┐  ┌────────────▼───────────────┐
   │  MedicalVisionClassifier│  │      MedicalRAG             │
   │  (PyTorch + Grad-CAM)  │  │  (Vector Store + TF-IDF)   │
   │  ResNet / Custom CNN   │  │  Clinical Guidelines Corpus │
   └───────────┬──────────┘  └────────────┬───────────────┘
               │                          │
               └────────────┬─────────────┘
                            │
              ┌─────────────▼──────────────┐
              │   ClinicalReportGenerator   │
              │   (RAG-augmented synthesis) │
              └─────────────────────────────┘
```

### Core Modules

| Module | Description |
|--------|-------------|
| `MedicalVisionClassifier` | PyTorch CNN for chest X-ray classification with Grad-CAM visual explainability |
| `MedicalRAG` | Vector store over clinical practice guidelines using TF-IDF / scikit-learn cosine similarity |
| `ClinicalReportGenerator` | Synthesises vision outputs with RAG-retrieved literature into structured medical reports |
| `Streamlit UI` | Interactive web dashboard — image upload, heatmap overlay, report generation, Q&A |

---

## Dataset

**NIH Chest X-Ray / Kaggle Chest X-Ray Images (Pneumonia)**

```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Download the dataset with:
```bash
python data/download_data.py
```

---

## Tech Stack

- **Framework:** PyTorch, torchvision
- **Explainability:** Grad-CAM (Class Activation Mapping)
- **RAG / NLP:** scikit-learn TF-IDF + cosine similarity (zero-dependency fallback available)
- **UI:** Streamlit
- **Visualisation:** Matplotlib, NumPy Jet colormap fallback
- **Image Processing:** Pillow (PIL)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Hadifarrag/Med_vision_Agent.git
cd Med_vision_Agent

# Install dependencies
pip install torch torchvision streamlit pillow scikit-learn matplotlib numpy

# (Optional) Download the dataset
python data/download_data.py
```

---

## Usage

### Launch the Streamlit App

```bash
streamlit run med.py
```

Open `http://localhost:8501` in your browser.

### CLI Verification Mode

```bash
python med.py --cli
```

---

## Features

- 🔬 **Chest X-Ray Classification** — Binary classification (Normal / Pneumonia) with softmax confidence scores
- 🌡️ **Grad-CAM Heatmaps** — Visual explainability showing which regions drove the prediction
- 📋 **AI Clinical Reports** — Structured reports combining imaging findings with evidence-based clinical literature
- 💬 **RAG Q&A** — Ask follow-up clinical questions; the system retrieves relevant guideline excerpts
- 🛡️ **Graceful Fallbacks** — Pure Python TF-IDF and NumPy colormap fallbacks when optional dependencies are unavailable

---

## Project Structure

```
Med_vision_Agent/
├── med.py              # Main application (Streamlit UI + all modules)
├── med.ipynb           # Jupyter notebook for experimentation
├── data/
│   ├── download_data.py
│   └── chest_xray/     # Dataset (train/val/test)
├── models/
│   └── datasets.py     # Dataset loaders
├── notebooks/
│   └── eda.py          # Exploratory data analysis
├── evals/              # Evaluation scripts
├── mcp_servers/        # MCP server integrations (experimental)
└── app/                # App assets
```

---

## Results

| Metric | Score |
|--------|-------|
| Task | Binary Classification (Normal / Pneumonia) |
| Architecture | PyTorch CNN + Grad-CAM |
| Explainability | Class Activation Maps (Grad-CAM) |
| RAG Source | Clinical Practice Guidelines corpus |

---

## Author

**Hady Mohamed Farrag**  
AI & Data Science Professional  
📧 hadyfarrag056@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/hady-farrag-6306061b6/)  
🌐 [Portfolio](https://hadifarrag.github.io)

---

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
