"""LoRA adapter exporter.

Exports only the LoRA adapters (small files) instead of the full model,
making it easy to share and deploy fine-tuned models efficiently.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


class LoRAExporter:
    """Export LoRA adapters separately from base model.

    LoRA adapters are typically very small (MB range) compared to full models (GB range),
    making them ideal for sharing and version control.

    Args:
        output_dir: Directory to save LoRA adapters
        adapter_name: Name for the adapter
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        adapter_name: Optional[str] = None
    ):
        """Initialize LoRA exporter."""
        self.output_dir = Path(output_dir)
        self.adapter_name = adapter_name or "lora_adapters"

        logger.info(f"LoRA exporter initialized: {self.output_dir}")

    def export_adapters(
        self,
        model: Any,
        base_model_name: str,
        tokenizer: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Export LoRA adapters only.

        Args:
            model: PEFT model with LoRA adapters
            base_model_name: Name/path of the base model
            tokenizer: Optional tokenizer (recommended to include)
            metadata: Optional training metadata

        Returns:
            Path to exported adapters

        Example:
            >>> exporter = LoRAExporter("./lora_adapters/my_adapter")
            >>> exporter.export_adapters(
            ...     model=peft_model,
            ...     base_model_name="liuhaotian/llava-v1.5-7b"
            ... )
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting LoRA adapters to {self.output_dir}...")

        # Check if model is a PEFT model
        if not hasattr(model, 'save_pretrained'):
            raise ValueError("Model must be a PEFT model with LoRA adapters")

        # Save LoRA adapters
        logger.info("  Saving LoRA adapter weights...")
        try:
            model.save_pretrained(self.output_dir)
            logger.info("  ✓ Adapters saved")
        except Exception as e:
            logger.error(f"Failed to save adapters: {e}")
            raise

        # Save tokenizer if provided
        if tokenizer:
            logger.info("  Saving tokenizer...")
            tokenizer.save_pretrained(self.output_dir)
            logger.info("  ✓ Tokenizer saved")

        # Create adapter info file
        adapter_info = {
            "base_model": base_model_name,
            "adapter_name": self.adapter_name,
            "adapter_type": "LoRA"
        }

        if metadata:
            adapter_info["training_metadata"] = metadata

        info_path = self.output_dir / "adapter_info.json"
        with open(info_path, 'w') as f:
            json.dump(adapter_info, f, indent=2)
        logger.info("  ✓ Adapter info saved")

        # Create README
        self._create_readme(base_model_name, metadata)

        # Calculate adapter size
        total_size = sum(f.stat().st_size for f in self.output_dir.rglob("*") if f.is_file())
        size_mb = total_size / 1024 / 1024

        logger.info("=" * 80)
        logger.info("✅ LoRA adapters exported successfully!")
        logger.info("=" * 80)
        logger.info(f"Location: {self.output_dir}")
        logger.info(f"Total size: {size_mb:.1f} MB")
        logger.info(f"Base model: {base_model_name}")
        logger.info("Files:")
        for file in sorted(self.output_dir.iterdir()):
            if file.is_file():
                file_size = file.stat().st_size / 1024 / 1024
                logger.info(f"  - {file.name} ({file_size:.2f} MB)")
        logger.info("=" * 80)

        return self.output_dir

    def load_and_merge(
        self,
        base_model_path: str,
        adapter_path: Optional[Path] = None
    ) -> Any:
        """Load adapters and merge with base model.

        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapters (default: self.output_dir)

        Returns:
            Merged model

        Example:
            >>> merged_model = exporter.load_and_merge(
            ...     base_model_path="liuhaotian/llava-v1.5-7b"
            ... )
        """
        adapter_path = adapter_path or self.output_dir

        logger.info(f"Loading base model from {base_model_path}...")
        try:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            logger.info("  ✓ Base model loaded")

            # Load and apply adapters
            logger.info(f"Loading adapters from {adapter_path}...")
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path
            )
            logger.info("  ✓ Adapters loaded")

            # Merge adapters
            logger.info("Merging adapters with base model...")
            merged_model = model.merge_and_unload()
            logger.info("  ✓ Model merged")

            return merged_model

        except Exception as e:
            logger.error(f"Failed to load and merge: {e}")
            raise

    def _create_readme(
        self,
        base_model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create README for LoRA adapters.

        Args:
            base_model_name: Base model name
            metadata: Optional metadata
        """
        readme_path = self.output_dir / "README.md"

        readme_content = f"""# {self.adapter_name}

LoRA adapters for fine-tuned vision-language model.

## Base Model

This adapter is designed to work with: **{base_model_name}**

## Adapter Information

- **Type**: LoRA (Low-Rank Adaptation)
- **Task**: Medical image segmentation
- **Framework**: PEFT (Parameter-Efficient Fine-Tuning)

"""

        if metadata:
            readme_content += "## Training Information\n\n"

            if "epochs" in metadata:
                readme_content += f"- **Epochs**: {metadata['epochs']}\n"
            if "final_loss" in metadata:
                readme_content += f"- **Final Loss**: {metadata['final_loss']:.4f}\n"
            if "lora_rank" in metadata:
                readme_content += f"- **LoRA Rank**: {metadata['lora_rank']}\n"
            if "lora_alpha" in metadata:
                readme_content += f"- **LoRA Alpha**: {metadata['lora_alpha']}\n"

            readme_content += "\n"

        readme_content += f"""## Usage

### Option 1: Load and merge with base model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model_name}",
    trust_remote_code=True
)

# Load adapters
model = PeftModel.from_pretrained(
    base_model,
    "path/to/this/directory"
)

# Merge adapters (for deployment)
merged_model = model.merge_and_unload()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "path/to/this/directory",  # or use base model path
    trust_remote_code=True
)
```

### Option 2: Use adapters directly (more memory efficient)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model in 4-bit (for inference on limited GPU)
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model_name}",
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

# Load adapters
model = PeftModel.from_pretrained(
    base_model,
    "path/to/this/directory"
)

# Now use model for inference
```

## Files

- `adapter_config.json`: LoRA adapter configuration
- `adapter_model.bin`: LoRA adapter weights (very small!)
- `adapter_info.json`: Information about base model and training
- `README.md`: This file

## Size Advantage

LoRA adapters are typically:
- **Full model**: Several GB (e.g., 13.5 GB for LLaVA-7B)
- **LoRA adapters**: Only ~10-50 MB! 📦

This makes them perfect for:
- Version control (Git)
- Quick sharing and deployment
- Fine-tuning multiple tasks (swap adapters)

## Citation

```bibtex
@misc{{idoctor-lora-adapter,
  title={{{{LoRA Adapter for Medical Image Segmentation}}}},
  author={{{{IDoctor Team}}}},
  year={{{{2025}}}},
  publisher={{{{IDoctor Platform}}}}
}}
```
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.info("  ✓ README created")


def export_lora_adapters(
    model: Any,
    base_model_name: str,
    output_dir: Union[str, Path],
    adapter_name: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """Convenience function to export LoRA adapters.

    Args:
        model: PEFT model
        base_model_name: Base model name/path
        output_dir: Output directory
        adapter_name: Adapter name
        tokenizer: Optional tokenizer
        metadata: Optional metadata

    Returns:
        Path to exported adapters

    Example:
        >>> from core.export import export_lora_adapters
        >>> export_lora_adapters(
        ...     model=peft_model,
        ...     base_model_name="liuhaotian/llava-v1.5-7b",
        ...     output_dir="./my_lora_adapter",
        ...     metadata={"epochs": 3, "lora_rank": 8}
        ... )
    """
    exporter = LoRAExporter(output_dir, adapter_name)
    return exporter.export_adapters(model, base_model_name, tokenizer, metadata)
