"""Model card generator.

Auto-generates comprehensive model cards with training details,
performance metrics, and usage examples.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """Generate model cards for exported models.

    Creates detailed model cards following best practices for
    model documentation and transparency.

    Args:
        model_name: Name of the model
        output_path: Path to save model card
    """

    def __init__(
        self,
        model_name: str,
        output_path: Union[str, Path]
    ):
        """Initialize model card generator."""
        self.model_name = model_name
        self.output_path = Path(output_path)

        logger.info(f"Model card generator initialized: {self.model_name}")

    def generate(
        self,
        base_model: str,
        training_data: Dict[str, Any],
        training_config: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None,
        limitations: Optional[List[str]] = None,
        ethical_considerations: Optional[List[str]] = None,
        usage_examples: Optional[List[Dict[str, str]]] = None
    ) -> Path:
        """Generate a comprehensive model card.

        Args:
            base_model: Base model name
            training_data: Training dataset information
            training_config: Training configuration details
            performance_metrics: Model performance metrics
            limitations: Known limitations
            ethical_considerations: Ethical considerations
            usage_examples: Code examples for using the model

        Returns:
            Path to generated model card

        Example:
            >>> generator = ModelCardGenerator("my-model", "./MODEL_CARD.md")
            >>> generator.generate(
            ...     base_model="liuhaotian/llava-v1.5-7b",
            ...     training_data={"samples": 1000, "domain": "medical"},
            ...     training_config={"method": "qlora", "epochs": 3}
            ... )
        """
        logger.info("Generating model card...")

        card_content = self._create_header()
        card_content += self._create_model_description(base_model)
        card_content += self._create_training_section(training_data, training_config)

        if performance_metrics:
            card_content += self._create_performance_section(performance_metrics)

        card_content += self._create_usage_section(usage_examples)

        if limitations:
            card_content += self._create_limitations_section(limitations)

        if ethical_considerations:
            card_content += self._create_ethical_section(ethical_considerations)

        card_content += self._create_citation_section()
        card_content += self._create_footer()

        # Save to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write(card_content)

        logger.info(f"✓ Model card generated: {self.output_path}")

        return self.output_path

    def _create_header(self) -> str:
        """Create model card header."""
        return f"""---
language: en
license: mit
tags:
- vision-language
- medical-imaging
- segmentation
- lora
- fine-tuned
library_name: transformers
pipeline_tag: image-to-text
---

# {self.model_name}

<div align="center">
  <img src="https://img.shields.io/badge/Model-Vision--Language-blue" alt="Model Type"/>
  <img src="https://img.shields.io/badge/Task-Medical%20Segmentation-green" alt="Task"/>
  <img src="https://img.shields.io/badge/Framework-Transformers-orange" alt="Framework"/>
</div>

"""

    def _create_model_description(self, base_model: str) -> str:
        """Create model description section."""
        return f"""## Model Description

{self.model_name} is a fine-tuned vision-language model specialized for medical image segmentation tasks.
The model is based on **{base_model}** and has been adapted using LoRA (Low-Rank Adaptation) for
parameter-efficient fine-tuning.

### Key Features

- 🏥 **Medical Image Understanding**: Trained on medical imaging data with expert annotations
- 🎯 **Precise Segmentation**: Capable of identifying and segmenting anatomical structures and abnormalities
- 💾 **Memory Efficient**: Fine-tuned using QLoRA (4-bit quantization) for deployment on consumer GPUs
- 🚀 **Fast Inference**: Optimized for real-time medical image analysis
- 🔬 **Explainable**: Provides natural language descriptions alongside segmentations

"""

    def _create_training_section(
        self,
        training_data: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> str:
        """Create training details section."""
        section = """## Training Details

### Training Data

"""

        # Training data details
        if "samples" in training_data:
            section += f"- **Number of Samples**: {training_data['samples']:,}\n"
        if "domain" in training_data:
            section += f"- **Medical Domain**: {training_data['domain']}\n"
        if "modality" in training_data:
            section += f"- **Imaging Modality**: {training_data['modality']}\n"
        if "source" in training_data:
            section += f"- **Data Source**: {training_data['source']}\n"

        section += "\n### Training Configuration\n\n"

        # Training config details
        if "method" in training_config:
            section += f"- **Training Method**: {training_config['method'].upper()}\n"
        if "epochs" in training_config:
            section += f"- **Epochs**: {training_config['epochs']}\n"
        if "batch_size" in training_config:
            section += f"- **Batch Size**: {training_config['batch_size']}\n"
        if "learning_rate" in training_config:
            section += f"- **Learning Rate**: {training_config['learning_rate']}\n"
        if "lora_rank" in training_config:
            section += f"- **LoRA Rank**: {training_config['lora_rank']}\n"
        if "lora_alpha" in training_config:
            section += f"- **LoRA Alpha**: {training_config['lora_alpha']}\n"

        section += "\n### Training Infrastructure\n\n"

        if "gpu" in training_config:
            section += f"- **GPU**: {training_config['gpu']}\n"
        if "mixed_precision" in training_config:
            section += f"- **Mixed Precision**: {training_config['mixed_precision']}\n"
        if "gradient_checkpointing" in training_config:
            section += f"- **Gradient Checkpointing**: {training_config['gradient_checkpointing']}\n"

        section += "\n"

        return section

    def _create_performance_section(self, metrics: Dict[str, float]) -> str:
        """Create performance metrics section."""
        section = """## Performance Metrics

| Metric | Value |
|--------|-------|
"""

        for metric_name, value in metrics.items():
            # Format metric name nicely
            formatted_name = metric_name.replace("_", " ").title()

            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            section += f"| {formatted_name} | {formatted_value} |\n"

        section += "\n"

        return section

    def _create_usage_section(
        self,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Create usage examples section."""
        section = """## Usage

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "path/to/model",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "path/to/model",
    trust_remote_code=True
)

# Load image
image = Image.open("medical_image.jpg")

# Prepare input
prompt = "Segment all anatomical structures in this medical image."
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate segmentation
outputs = model.generate(**inputs, max_length=512)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
```

### Memory-Efficient Usage (8GB GPU)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "path/to/model",
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Now use model for inference with only ~4GB VRAM!
```

"""

        # Add custom examples if provided
        if examples:
            section += "### Additional Examples\n\n"
            for i, example in enumerate(examples, 1):
                title = example.get("title", f"Example {i}")
                code = example.get("code", "")

                section += f"#### {title}\n\n```python\n{code}\n```\n\n"

        return section

    def _create_limitations_section(self, limitations: List[str]) -> str:
        """Create limitations section."""
        section = """## Limitations

"""
        for limitation in limitations:
            section += f"- {limitation}\n"

        section += "\n"

        return section

    def _create_ethical_section(self, considerations: List[str]) -> str:
        """Create ethical considerations section."""
        section = """## Ethical Considerations

"""
        for consideration in considerations:
            section += f"- {consideration}\n"

        section += "\n"

        return section

    def _create_citation_section(self) -> str:
        """Create citation section."""
        year = datetime.now().year

        return f"""## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{{self.model_name.lower().replace('-', '_').replace(' ', '_')},
  title={{{{{self.model_name}}}}},
  author={{{{IDoctor Team}}}},
  year={{{{{year}}}}},
  publisher={{{{IDoctor Platform}}}},
  howpublished={{{{\\url{{https://github.com/idoctor/sam2-training}}}}}}
}}
```

"""

    def _create_footer(self) -> str:
        """Create model card footer."""
        return f"""## Model Card Authors

This model card was generated automatically by the IDoctor training pipeline.

**Last Updated**: {datetime.now().strftime("%Y-%m-%d")}

---

For questions or issues, please contact the IDoctor team or open an issue on GitHub.
"""


def generate_model_card(
    model_name: str,
    output_path: Union[str, Path],
    base_model: str,
    training_data: Dict[str, Any],
    training_config: Dict[str, Any],
    performance_metrics: Optional[Dict[str, float]] = None,
    **kwargs
) -> Path:
    """Convenience function to generate a model card.

    Args:
        model_name: Model name
        output_path: Output path
        base_model: Base model name
        training_data: Training data info
        training_config: Training config
        performance_metrics: Performance metrics
        **kwargs: Additional arguments (limitations, ethical_considerations, etc.)

    Returns:
        Path to generated model card

    Example:
        >>> from core.export import generate_model_card
        >>> generate_model_card(
        ...     model_name="IDoctor-SAM2-Medical-7B",
        ...     output_path="./MODEL_CARD.md",
        ...     base_model="liuhaotian/llava-v1.5-7b",
        ...     training_data={"samples": 5000, "domain": "Radiology"},
        ...     training_config={"method": "qlora", "epochs": 3},
        ...     performance_metrics={"iou": 0.85, "dice": 0.89}
        ... )
    """
    generator = ModelCardGenerator(model_name, output_path)
    return generator.generate(
        base_model=base_model,
        training_data=training_data,
        training_config=training_config,
        performance_metrics=performance_metrics,
        **kwargs
    )
