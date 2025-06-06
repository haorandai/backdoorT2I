# BackdoorT2I: Text-to-Image Backdoor Attack Framework

A comprehensive framework for implementing backdoor attacks on text-to-image diffusion models.

## Overview

This project implements a framework for studying backdoor attacks on text-to-image diffusion models. It includes tools for generating adversarial perturbations, training backdoored models, and querying images with trigger patterns.

## Project Structure

### `src/` - Source Code Implementation

#### `GenSample/`
- Implementation of perturbation generation algorithms
- Example scripts for generating adversarial samples
- Utilities for trigger pattern creation and manipulation

#### `TrainingPhase/`
- Training scripts for various text-to-image models
- Implementation of backdoor injection during model training
- Modified versions of Hugging Face's official training scripts
- Custom training utilities for backdoored models

#### `ImageQuery/`
- Tools for querying backdoored models with trigger patterns
- Image generation and evaluation scripts
- Utilities for analyzing model responses to trigger inputs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/backdoorT2I.git
cd backdoorT2I

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generating Adversarial Samples
```python
from src.GenSample.perturbation_generator import PerturbationGenerator

# Initialize generator
generator = PerturbationGenerator()

# Generate adversarial samples
samples = generator.generate(trigger_pattern="your_trigger")
```

### Training Backdoored Models
```python
from src.TrainingPhase.trainer import BackdoorTrainer

# Initialize trainer
trainer = BackdoorTrainer(
    model_name="stabilityai/stable-diffusion-2-1",
    trigger_pattern="your_trigger"
)

# Start training
trainer.train()
```

### Querying Images
```python
from src.ImageQuery.query_engine import QueryEngine

# Initialize query engine
engine = QueryEngine(model_path="path_to_backdoored_model")

# Generate images with trigger
images = engine.generate(prompt="your_prompt", trigger="your_trigger")
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Diffusers
- Transformers
- CUDA-compatible GPU (recommended)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{backdoorT2I,
  author = {Your Name},
  title = {BackdoorT2I: Text-to-Image Backdoor Attack Framework},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/backdoorT2I}
}
```