# T2I Backdoor Attacks Impl

 Implementation of backdoor attacks on text-to-image diffusion models.

## Overview

This project implements a framework for studying backdoor attacks on text-to-image diffusion models. It includes tools for generating adversarial perturbations, training backdoored models, and querying images with trigger patterns.

## Project Structure

### `src/` - Source Code Implementation

#### `GenSample/`
- Implementation of perturbation generation algorithms
- Example scripts for generating adversarial samples

#### `TrainingPhase/`
- Training scripts for various text-to-image models
- Implementation of backdoor injection during model training
- Modified versions of Hugging Face's official training scripts
- Custom training utilities for backdoored models

#### `ImageQuery/`
- Tools for querying backdoored models with trigger patterns
- Image generation scripts
