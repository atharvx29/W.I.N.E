# W.I.N.E. - Winely Intelligent Neural Entity

A neural network-based wine classification system built with PyTorch. Classifies wines into three classes based on 13 chemical features using a feedforward neural network.

> **Note:** This project was developed entirely without AI assistance using the Stack Overflow method as a personal challenge.

## Project Structure

```
wine-classification/
├── dataset.csv              # Training dataset
├── test_data.csv            # Test dataset
├── model_architecture.py    # Neural network model definition
├── train_test.py            # Training and evaluation script
├── main.py                  # Inference script
├── W.I.N.E..pth             # Trained model weights
└── scaler.pkl               # Fitted StandardScaler
```

## Model Architecture

- **Input Layer:** 13 features (chemical properties of wine)
- **Hidden Layer:** 16 neurons with ReLU activation
- **Output Layer:** 3 classes (wine varieties)
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 1000

## Requirements

- Python 3.x
- PyTorch
- pandas
- scikit-learn
- joblib

## Usage

### Training

```bash
python train_test.py
```

Trains the model on `dataset.csv`, saves weights to `W.I.N.E..pth` and scaler to `scaler.pkl`.

### Inference

```bash
python main.py
```

Loads the trained model and classifies wines from `test_data.csv`.

## Author

**Atharv Sharma**

Founder, NexSemble
