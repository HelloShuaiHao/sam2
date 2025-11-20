"""HuggingFace model exporter.

Exports trained models in HuggingFace format for easy sharing and deployment.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json

logger = logging.getLogger(__name__)


class HuggingFaceExporter:
    """Export models in HuggingFace format.

    Exports complete models with tokenizer, config, and all necessary files
    in a format compatible with HuggingFace's `from_pretrained()` method.

    Args:
        output_dir: Directory to save exported model
        model_name: Optional model name for the export
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        model_name: Optional[str] = None
    ):
        """Initialize HuggingFace exporter."""
        self.output_dir = Path(output_dir)
        self.model_name = model_name or "exported_model"

        logger.info(f"HuggingFace exporter initialized: {self.output_dir}")

    def export_model(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Export a complete model in HuggingFace format.

        Args:
            model: PyTorch model (HuggingFace model or PEFT model)
            tokenizer: HuggingFace tokenizer
            config: Optional additional configuration
            metadata: Optional metadata (training info, metrics, etc.)

        Returns:
            Path to exported model directory

        Example:
            >>> exporter = HuggingFaceExporter("./exported_models/my_model")
            >>> exporter.export_model(model, tokenizer, metadata={"epochs": 3})
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to {self.output_dir}...")

        # Save model
        logger.info("  Saving model weights...")
        try:
            # Check if it's a PEFT model
            if hasattr(model, 'merge_and_unload'):
                logger.info("  Detected PEFT model - merging adapters...")
                # For LoRA models, we can either save adapters only or merge
                # Here we save the merged model for easier deployment
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(self.output_dir)
                logger.info("  ✓ Merged model saved")
            else:
                # Regular model
                model.save_pretrained(self.output_dir)
                logger.info("  ✓ Model saved")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

        # Save tokenizer
        logger.info("  Saving tokenizer...")
        try:
            tokenizer.save_pretrained(self.output_dir)
            logger.info("  ✓ Tokenizer saved")
        except Exception as e:
            logger.error(f"Failed to save tokenizer: {e}")
            raise

        # Save additional config if provided
        if config:
            config_path = self.output_dir / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("  ✓ Training config saved")

        # Save metadata
        if metadata:
            metadata_path = self.output_dir / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info("  ✓ Metadata saved")

        # Create README
        self._create_readme(metadata)

        logger.info("=" * 80)
        logger.info("✅ Model exported successfully!")
        logger.info("=" * 80)
        logger.info(f"Location: {self.output_dir}")
        logger.info(f"Files:")
        for file in sorted(self.output_dir.iterdir()):
            logger.info(f"  - {file.name}")
        logger.info("=" * 80)

        return self.output_dir

    def verify_export(self) -> bool:
        """Verify that the exported model can be loaded.

        Returns:
            True if model can be loaded successfully

        Example:
            >>> exporter.export_model(model, tokenizer)
            >>> if exporter.verify_export():
            ...     print("Export verified!")
        """
        logger.info("Verifying exported model...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Try to load model
            logger.info("  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.output_dir,
                trust_remote_code=True
            )
            logger.info("  ✓ Model loaded")

            # Try to load tokenizer
            logger.info("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.output_dir,
                trust_remote_code=True
            )
            logger.info("  ✓ Tokenizer loaded")

            logger.info("✅ Export verification passed!")
            return True

        except Exception as e:
            logger.error(f"❌ Export verification failed: {e}")
            return False

    def _create_readme(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a README file for the exported model.

        Args:
            metadata: Optional training metadata
        """
        readme_path = self.output_dir / "README.md"

        readme_content = f"""# {self.model_name}

This model was exported from the IDoctor SAM2 training pipeline.

## Model Description

This is a vision-language model fine-tuned for medical image segmentation tasks.

"""

        if metadata:
            readme_content += "## Training Information\n\n"

            if "epochs" in metadata:
                readme_content += f"- **Epochs**: {metadata['epochs']}\n"
            if "final_loss" in metadata:
                readme_content += f"- **Final Loss**: {metadata['final_loss']:.4f}\n"
            if "dataset_size" in metadata:
                readme_content += f"- **Dataset Size**: {metadata['dataset_size']}\n"

            readme_content += "\n"

        readme_content += """## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "path/to/this/directory",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "path/to/this/directory",
    trust_remote_code=True
)

# Use for inference
# (See inference example for more details)
```

## Files

- `pytorch_model.bin` or `model.safetensors`: Model weights
- `config.json`: Model configuration
- `tokenizer_config.json`: Tokenizer configuration
- `training_config.json`: Training configuration (if available)
- `training_metadata.json`: Training metadata (if available)

## Citation

If you use this model, please cite:

```bibtex
@misc{idoctor-sam2-finetuned,
  title={Fine-tuned Vision-Language Model for Medical Image Segmentation},
  author={IDoctor Team},
  year={2025},
  publisher={IDoctor Platform}
}
```
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.info("  ✓ README created")

    def create_model_archive(
        self,
        archive_name: Optional[str] = None,
        format: str = "tar.gz"
    ) -> Path:
        """Create a compressed archive of the exported model.

        Args:
            archive_name: Name of the archive (without extension)
            format: Archive format ('tar.gz', 'zip')

        Returns:
            Path to created archive

        Example:
            >>> exporter.export_model(model, tokenizer)
            >>> archive_path = exporter.create_model_archive("my_model_v1")
        """
        if not self.output_dir.exists():
            raise ValueError("No model has been exported yet")

        archive_name = archive_name or self.model_name
        logger.info(f"Creating model archive: {archive_name}.{format}")

        if format == "tar.gz":
            import tarfile
            archive_path = self.output_dir.parent / f"{archive_name}.tar.gz"

            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(self.output_dir, arcname=self.model_name)

        elif format == "zip":
            import zipfile
            archive_path = self.output_dir.parent / f"{archive_name}.zip"

            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file in self.output_dir.rglob("*"):
                    if file.is_file():
                        arcname = file.relative_to(self.output_dir.parent)
                        zipf.write(file, arcname)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"✓ Archive created: {archive_path}")
        logger.info(f"  Size: {archive_path.stat().st_size / 1024 / 1024:.1f} MB")

        return archive_path


def export_huggingface_model(
    model: Any,
    tokenizer: Any,
    output_dir: Union[str, Path],
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    verify: bool = True
) -> Path:
    """Convenience function to export a model in HuggingFace format.

    Args:
        model: Model to export
        tokenizer: Tokenizer to export
        output_dir: Output directory
        model_name: Optional model name
        config: Optional configuration
        metadata: Optional metadata
        verify: Whether to verify the export

    Returns:
        Path to exported model

    Example:
        >>> from core.export import export_huggingface_model
        >>> export_path = export_huggingface_model(
        ...     model=trained_model,
        ...     tokenizer=tokenizer,
        ...     output_dir="./my_model",
        ...     metadata={"epochs": 3, "final_loss": 0.234}
        ... )
    """
    exporter = HuggingFaceExporter(output_dir, model_name)
    export_path = exporter.export_model(model, tokenizer, config, metadata)

    if verify:
        exporter.verify_export()

    return export_path
