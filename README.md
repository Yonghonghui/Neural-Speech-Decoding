# Neural Speech Decoding: Brain-to-Text System

**UCLA ECE 243 Final Project (Fall 2025)**

This repository contains the implementation of a real-time neural speech decoder designed to predict phonemes from intracranial neural recordings (area 6v). The project builds upon a GRU baseline and systematically optimizes the pipeline using advanced architectures (LSTM), loss functions (Focal CTC), and regularization strategies.

## 📖 Project Overview

Brain-Computer Interfaces (BCIs) aim to restore communication for individuals with severe speech deficits. This project utilizes the **Brain-to-Text Benchmark '24** dataset. The core challenge is to minimize the Phoneme Error Rate (PER) while strictly adhering to real-time, uni-directional processing constraints.

Our best model achieves a **PER of 19.59%**, significantly outperforming the baseline of 23.6%.

📄 **Full Report:** [ECE243_FinalProject_report.pdf](./ECE243_FinalProject_report.pdf)

## 📊 Key Results

Summary of experimental results comparing our optimized LSTM approach against the baseline and other architectures.

| Experiment | Method | Hardware | Val PER | Time/Batch |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | GRU + Adam | T4 | 23.6% | ~0.76s |
| Exp 2 | GRU + Linderman Arch | T4 | 21.9% | ~0.76s |
| Exp 5 | + Focal CTC Loss | T4 | 21.47% | ~0.77s |
| **Exp 6 (Best)** | **LSTM + Linderman Stack** | **T4** | **19.59%** | **~0.88s** |
| Exp 7 | Transformer (Causal) | T4 | 28.0% | ~0.29s |
| Exp 8 | High-Cap GRU (1024u) | 5070 Ti | 19.65% | ~0.20s* |

*\*Time/Batch for Exp 8 reflects acceleration from newer hardware (RTX 5070 Ti).*

## 📂 Repository Structure

The project is organized as follows:

```text
Neural-Speech-Decoding/
├── neural_seq_decoder/
│   ├── src/neural_decoder/       # Core library code
│   │   ├── model.py              # LSTM, GRU, and Transformer architectures
│   │   ├── losses.py             # Focal CTC Loss implementation
│   │   ├── dataset.py            # Data loading and preprocessing
│   │   ├── augmentations.py      # Time masking and noise injection
│   │   └── neural_decoder_trainer.py # Training loop and scheduler
│   ├── scripts/                  # Executable scripts
│   │   ├── train_model.py        # Main entry point for training
│   │   └── eval_competition.py   # Evaluation script
│   ├── notebooks/                # Jupyter notebooks for data formatting
│   └── logs/                     # Training logs and checkpoints
├── pyproject.toml                # Build system configuration
├── setup.py                      # Package setup
└── README.md

```
## 💾 Dataset and Pre-trained Models

Due to storage limits, the dataset and pre-trained models are hosted externally on Google Drive.

[📥 **Download Dataset & Models Here**](https://drive.google.com/drive/folders/1QUhap4d-6aJLZPaJmYvh7jHOnLH1z7Wb?usp=sharing)

### Setup Instructions

1.  Download the contents from the link above.
2.  **Unzip and Place Files**:
    * Place the `competitionData` folder **inside** the `neural_seq_decoder` directory.
    * Place the `ptDecoder_ctc.pkl` file **inside** the `neural_seq_decoder` directory.

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.8+
* PyTorch (CUDA version recommended for GPU training)

### 2. Installation

Clone the repository and install the package in editable mode:

```bash
git clone [https://github.com/Yonghonghui/Neural-Speech-Decoding.git](https://github.com/Yonghonghui/Neural-Speech-Decoding.git)
cd Neural-Speech-Decoding
pip install -e .

```

### 3. Training the Model

To train the optimal LSTM model (Exp 6), use the `train_model.py` script. You may need to adjust the dataset path in the arguments.

```bash
python neural_seq_decoder/scripts/train_model.py \
    --datasetPath /path/to/competitionData \
    --outputDir ./logs/my_experiment \
    --nUnits 256 \
    --nLayers 5 \
    --lrStart 0.001 \
    --dropout 0.2 \
    --whiteNoiseSD 1.0

```

## 🧠 Methodology Highlights

* **Architecture Search:** We transitioned from a vanilla GRU to an **LSTM backbone** to better capture long-range temporal dependencies in neural signals.
* **Linderman Stack:** We implemented a post-RNN processing stack consisting of `Linear -> LayerNorm -> Dropout -> GELU` to increase representational capacity and training stability.
* **Focal CTC Loss:** To address class imbalance between common and rare phonemes, we implemented a Focal variant of the CTC loss function.
* **Optimization:** We utilized **AdamW** with a sequential scheduler (Linear Warmup + Cosine Annealing) to ensure convergence.

## 👥 Authors

* **Yanghonghui Chen**
* **Mu Li**
* **Kaiwen Zhao**

Dept. of Electrical and Computer Engineering, UCLA.

---

*Note: The dataset used in this project is part of the Brain-to-Text Benchmark '24 and is not included in this repository due to size/license constraints.*

