"""
Setup script for integrating HIT Fusion Model into existing pipeline.
This script creates the necessary directory structure and moves files to the right locations.
"""

import os
import shutil
from pathlib import Path


class FusionModelSetup:
    """Setup manager for HIT Fusion Model integration."""

    def __init__(self, project_root: str):
        """
        Args:
            project_root: Root directory of your project
        """
        self.root = Path(project_root)
        self.elements_dir = self.root / 'elements'
        self.models_dir = self.root / 'models'
        self.log_dir = self.root / 'log'
        self.dataset_dir = self.root / 'dataset'

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        print("Creating directories...")

        dirs_to_create = [
            self.elements_dir,
            self.src_models_dir,
            self.models_dir / 'fusion',
            self.log_dir / 'fusion_experiment',
            self.dataset_dir
        ]

        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {directory}")

    def copy_fusion_files(self):
        """Copy fusion model files to appropriate locations."""
        print("\nCopying fusion model files...")

        # Files to copy to elements/
        elements_files = [
            'fusion_model.py',
            'load_fusion_data.py'
        ]

        # Files to copy to src/models/
        src_files = [
            'hit_fusion_pipeline.py'
        ]

        current_dir = Path.cwd()

        for filename in elements_files:
            src_file = current_dir / filename
            if src_file.exists():
                dest = self.elements_dir / filename
                shutil.copy2(src_file, dest)
                print(f"  ✓ Copied {filename} to elements/")
            else:
                print(f"  ⚠ {filename} not found in current directory")

        for filename in src_files:
            src_file = current_dir / filename
            if src_file.exists():
                dest = self.src_models_dir / filename
                shutil.copy2(src_file, dest)
                print(f"  ✓ Copied {filename} to src/models/")
            else:
                print(f"  ⚠ {filename} not found in current directory")

    def create_init_files(self):
        """Create __init__.py files for Python packages."""
        print("\nCreating __init__.py files...")

        init_locations = [
            self.elements_dir,
            self.src_models_dir
        ]

        for location in init_locations:
            init_file = location / '__init__.py'
            if not init_file.exists():
                init_file.touch()
                print(f"  ✓ Created {init_file}")

    def update_imports(self):
        """Create a helper file for import management."""
        print("\nCreating import helper...")

        import_helper = self.root / 'fusion_imports.py'

        content = '''"""
Helper file for importing fusion model components.
Add this to your Python path or import from here.
"""

import sys
from pathlib import Path

# Add elements directory to path
project_root = Path(__file__).parent
elements_path = project_root / 'elements'
sys.path.insert(0, str(elements_path))

# Now you can import fusion components
from fusion_model import HITFusionModel, create_fusion_model
from load_fusion_data import FusionDataset, create_fusion_dataloaders
from hit_fusion_pipeline import FusionConfig, FusionTrainer, run_fusion_pipeline

__all__ = [
    'HITFusionModel',
    'create_fusion_model',
    'FusionDataset',
    'create_fusion_dataloaders',
    'FusionConfig',
    'FusionTrainer',
    'run_fusion_pipeline'
]
'''

        with open(import_helper, 'w') as f:
            f.write(content)

        print(f"  ✓ Created fusion_imports.py")

    def create_requirements(self):
        """Create/update requirements.txt with fusion model dependencies."""
        print("\nUpdating requirements...")

        requirements_file = self.root / 'requirements_fusion.txt'

        fusion_requirements = [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
            'numpy>=1.19.0',
            'opencv-python>=4.5.0',
            'matplotlib>=3.3.0',
            'tensorboard>=2.5.0',
            'scikit-learn>=0.24.0',
            'pandas>=1.2.0',
            'Pillow>=8.0.0',
            'tqdm>=4.60.0'
        ]

        with open(requirements_file, 'w') as f:
            f.write('\n'.join(fusion_requirements))

        print(f"  ✓ Created requirements_fusion.txt")

    def create_example_script(self):
        """Create an example training script."""
        print("\nCreating example script...")

        example_script = self.root / 'train_fusion_example.py'

        content = '''#!/usr/bin/env python3
"""
Example script for training the HIT Fusion Model.
Adjust paths and parameters as needed for your dataset.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import fusion components
from elements.fusion_model import create_fusion_model
from src.models.hit_fusion_pipeline import FusionConfig, FusionTrainer
from elements.utils import LoggerSingleton

def main():
    # Setup logging
    working_dir = os.getcwd()
    LoggerSingleton.setup_logger(working_dir)
    logger = LoggerSingleton.get_logger()

    # Create configuration
    config = FusionConfig()

    # Adjust these paths for your setup
    config.train_dataset_path = 'dataset/train_tiled.npz'
    config.test_dataset_path = 'dataset/test.npz'

    # Optional: Use pretrained encoders
    # config.use_pretrained = True
    # config.pretrained_cnn1d = 'log/cnn1d_best/model.npz'
    # config.pretrained_cnn3d = 'log/cnn3d_best/model.npz'
    # config.freeze_encoders = True  # Freeze during initial training

    # Training settings
    config.batch_size = 32
    config.num_epochs = 100
    config.learning_rate = 0.001
    config.patch_size = 9

    # Pipeline control
    config.do_train = True
    config.do_test = False

    # Create trainer and run
    logger.info("Starting HIT Fusion Model training...")
    trainer = FusionTrainer(config)

    if config.do_train:
        trainer.train()

    if config.do_test:
        trainer.test()

    logger.info("Complete!")

if __name__ == '__main__':
    main()
'''

        with open(example_script, 'w') as f:
            f.write(content)

        os.chmod(example_script, 0o755)  # Make executable
        print(f"  ✓ Created train_fusion_example.py")

    def create_readme(self):
        """Create README for fusion model."""
        print("\nCreating README...")

        readme = self.root / 'FUSION_MODEL_README.md'

        content = '''# HIT Fusion Model Integration

This directory contains the integrated HIT Fusion Model for hyperspectral image classification.

## Architecture

The fusion model combines two expert branches:

1. **Spectral Branch (CNN1D)**: Processes the 224-band spectral signature
2. **Spatial Branch (CNN3D/UNet)**: Processes spatial context in 2D/3D tiles

These branches are fused using cross-modal attention mechanisms to create a unified
representation that captures both spectral and spatial information.

## Directory Structure

```
├── elements/
│   ├── fusion_model.py          # Fusion model architecture
│   ├── load_fusion_data.py      # Data loading for fusion
│   └── [existing elements]
├── src/
│   └── models/
│       └── hit_fusion_pipeline.py  # Training pipeline
├── models/
│   └── fusion/                  # Saved fusion models
├── log/
│   └── fusion_experiment/       # Training logs and tensorboard
├── dataset/
│   └── train_tiled.npz         # Tiled training data
└── train_fusion_example.py     # Example training script

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_fusion.txt
```

### 2. Prepare Data

Ensure you have tiled datasets created using `create_hs_dataset.py`:

```bash
python create_hs_dataset.py
```

This should create `train_tiled.npz` in your dataset directory.

### 3. Train the Model

#### Option A: Using the example script

```bash
python train_fusion_example.py
```

#### Option B: Custom training

```python
from src.models.hit_fusion_pipeline import FusionConfig, FusionTrainer

config = FusionConfig()
config.train_dataset_path = 'dataset/train_tiled.npz'
config.batch_size = 32
config.num_epochs = 100

trainer = FusionTrainer(config)
trainer.train()
```

### 4. Use Pretrained Encoders (Optional)

If you have pretrained CNN1D and CNN3D models:

```python
config.use_pretrained = True
config.pretrained_cnn1d = 'log/cnn1d_best/model.npz'
config.pretrained_cnn3d = 'log/cnn3d_best/model.npz'
config.freeze_encoders = True  # Start with frozen encoders
```

## Key Parameters

### Model Architecture

- `embed_size`: Embedding dimension (default: 256)
- `num_heads`: Number of attention heads (default: 4)
- `num_conv_layers`: Convolutional layers per branch (default: 3)
- `patch_size`: Spatial patch size (default: 9x9)

### Training

- `batch_size`: Batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 0.001)
- `num_epochs`: Training epochs (default: 100)
- `patience`: Scheduler patience (default: 10)

## Monitoring Training

View training progress with tensorboard:

```bash
tensorboard --logdir=log/fusion_experiment
```

## Model Outputs

The fusion model produces:

- **During Training**: Saved models in `log/fusion_experiment/model.npz`
- **During Testing**: Predictions in `log/fusion_experiment/test_results.npz`

## Advantages of Fusion

1. **Better Accuracy**: Combines spectral "what" with spatial "where"
2. **Robustness**: Handles ambiguous cases better than single-modality models
3. **Interpretability**: Attention weights show which modality dominates per decision
4. **Transfer Learning**: Can leverage pretrained expert models

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Reduce `patch_size`
- Enable `freeze_encoders` to reduce trainable parameters

### Slow Training

- Increase `num_workers` in data loaders
- Ensure GPU is being used (`config.device = 'cuda'`)
- Use mixed precision training (already enabled via GradScaler)

### Poor Performance

- Ensure datasets are properly preprocessed
- Try unfreezing encoders after initial training
- Adjust `learning_rate` and `embed_size`

## Citation

If you use this fusion model, please cite the original HIT project.

## Support

For questions or issues, refer to the main HIT documentation or contact the project maintainers.
'''

        with open(readme, 'w') as f:
            f.write(content)

        print(f"  ✓ Created FUSION_MODEL_README.md")

    def verify_existing_files(self):
        """Verify that required existing files are present."""
        print("\nVerifying existing HIT files...")

        required_files = [
            self.elements_dir / 'load_data.py',
            self.elements_dir / 'load_model.py',
            self.elements_dir / 'utils.py',
            self.elements_dir / 'calc_loss.py',
            self.elements_dir / 'save_model.py'
        ]

        all_present = True
        for file_path in required_files:
            if file_path.exists():
                print(f"  ✓ Found {file_path.name}")
            else:
                print(f"  ✗ Missing {file_path.name}")
                all_present = False

        return all_present

    def run_setup(self):
        """Run complete setup process."""
        print("=" * 80)
        print("HIT FUSION MODEL SETUP")
        print("=" * 80)
        print(f"\nProject root: {self.root}\n")

        # Verify existing files
        if not self.verify_existing_files():
            print("\n⚠ Warning: Some required HIT files are missing.")
            print("  Make sure you're running this in the correct directory.")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Setup cancelled.")
                return

        # Run setup steps
        self.create_directories()
        self.copy_fusion_files()
        self.create_init_files()
        self.update_imports()
        self.create_requirements()
        self.create_example_script()
        self.create_readme()

        print("\n" + "=" * 80)
        print("SETUP COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements_fusion.txt")
        print("2. Review FUSION_MODEL_README.md for usage instructions")
        print("3. Run example: python train_fusion_example.py")
        print("\nFor detailed documentation, see FUSION_MODEL_README.md")


def main():
    """Main setup function."""
    import sys

    # Get project root from command line or use current directory
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.getcwd()

    setup = FusionModelSetup(project_root)
    setup.run_setup()


if __name__ == '__main__':
    main()