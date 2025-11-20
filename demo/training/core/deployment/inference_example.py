"""Inference example for exported models.

Demonstrates how to load and use fine-tuned models for medical image segmentation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """Inference wrapper for fine-tuned vision-language models.

    Supports both full models and LoRA adapters with 4-bit quantization
    for memory-efficient inference.

    Example:
        >>> inferencer = ModelInference("./exported_model")
        >>> result = inferencer.predict(
        ...     image_path="medical_image.jpg",
        ...     prompt="Segment all anatomical structures"
        ... )
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        use_4bit: bool = True,
        device: str = "auto"
    ):
        """Initialize model for inference.

        Args:
            model_path: Path to exported model directory
            use_4bit: Use 4-bit quantization for memory efficiency
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_path = Path(model_path)
        self.use_4bit = use_4bit
        self.device = device

        logger.info(f"Loading model from {self.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Set up quantization if requested
        if self.use_4bit and torch.cuda.is_available():
            logger.info("Using 4-bit quantization...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                device_map=self.device,
                trust_remote_code=True
            )
        else:
            logger.info("Loading model in full precision...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != "auto" else None,
                trust_remote_code=True
            )

        self.model.eval()
        logger.info("✓ Model loaded successfully!")

        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"GPU memory: {allocated:.2f} GB")

    def predict(
        self,
        image_path: Union[str, Path],
        prompt: str = "Describe and segment all visible structures in this medical image.",
        max_length: int = 512,
        temperature: float = 0.7,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Run inference on an image.

        Args:
            image_path: Path to input image
            prompt: Text prompt for the model
            max_length: Maximum generation length
            temperature: Sampling temperature
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with prediction results

        Example:
            >>> result = inferencer.predict(
            ...     image_path="scan.jpg",
            ...     prompt="Identify and segment the tumor",
            ...     max_length=256
            ... )
            >>> print(result['text'])
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Loaded image: {image_path} (size: {image.size})")

        # Prepare input
        # Note: This is a simplified example. Actual preprocessing depends on the model.
        # For vision-language models, you typically need a processor.
        logger.info(f"Prompt: {prompt}")

        # For demonstration, we'll show the general pattern
        # You may need to adapt this based on your specific model

        try:
            # Format prompt (adjust based on your model's format)
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt"
            )

            # Move to device
            if torch.cuda.is_available() and self.device != "cpu":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            logger.info("Generating response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    **generation_kwargs
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract assistant's response
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text

            logger.info("✓ Generation complete!")

            return {
                "text": response,
                "full_output": generated_text,
                "prompt": prompt,
                "image_path": str(image_path),
                "image_size": image.size
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def batch_predict(
        self,
        image_paths: list,
        prompts: Optional[list] = None,
        **kwargs
    ) -> list:
        """Run inference on multiple images.

        Args:
            image_paths: List of image paths
            prompts: Optional list of prompts (one per image)
            **kwargs: Additional generation parameters

        Returns:
            List of prediction results
        """
        if prompts is None:
            prompts = ["Segment all visible structures."] * len(image_paths)

        results = []
        for img_path, prompt in zip(image_paths, prompts):
            result = self.predict(img_path, prompt, **kwargs)
            results.append(result)

        return results


# =============================================================================
# Example 1: Basic Usage
# =============================================================================

def example_basic():
    """Basic inference example."""
    print("=" * 80)
    print("Example 1: Basic Inference")
    print("=" * 80)

    # Initialize inferencer
    inferencer = ModelInference(
        model_path="./exported_model",  # Path to your exported model
        use_4bit=True  # Use 4-bit quantization for memory efficiency
    )

    # Run prediction
    result = inferencer.predict(
        image_path="./medical_image.jpg",
        prompt="Identify and segment the tumor region.",
        max_length=256
    )

    print(f"\nResult: {result['text']}")
    print()


# =============================================================================
# Example 2: Batch Inference
# =============================================================================

def example_batch():
    """Batch inference example."""
    print("=" * 80)
    print("Example 2: Batch Inference")
    print("=" * 80)

    inferencer = ModelInference("./exported_model")

    # Multiple images
    images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    prompts = [
        "Segment the heart",
        "Identify lung abnormalities",
        "Segment the liver"
    ]

    # Batch predict
    results = inferencer.batch_predict(images, prompts)

    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result['text'][:100]}...")


# =============================================================================
# Example 3: LoRA Adapter Loading
# =============================================================================

def example_lora():
    """Example of loading LoRA adapters separately."""
    print("=" * 80)
    print("Example 3: LoRA Adapter Loading")
    print("=" * 80)

    from peft import PeftModel

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "liuhaotian/llava-v1.5-7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        "./lora_adapters"  # Path to your LoRA adapters
    )

    # Optionally merge for faster inference
    model = model.merge_and_unload()

    print("✓ Model with LoRA adapters loaded!")


# =============================================================================
# Example 4: Memory-Efficient Inference (8GB GPU)
# =============================================================================

def example_memory_efficient():
    """Ultra memory-efficient inference for 8GB GPUs."""
    print("=" * 80)
    print("Example 4: Memory-Efficient Inference (8GB GPU)")
    print("=" * 80)

    # Very aggressive memory optimization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "./exported_model",
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Additional memory optimization
    )

    tokenizer = AutoTokenizer.from_pretrained("./exported_model")

    # Check memory
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    print("✓ Model loaded in ~4GB VRAM!")


# =============================================================================
# Example 5: REST API Deployment
# =============================================================================

def example_fastapi():
    """Example FastAPI server for model deployment."""
    print("=" * 80)
    print("Example 5: FastAPI Deployment")
    print("=" * 80)

    print("""
# Save this as inference_api.py:

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io

app = FastAPI(title="IDoctor Model API")

# Initialize model globally
inferencer = ModelInference("./exported_model", use_4bit=True)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    prompt: str = Form("Segment all visible structures")
):
    # Read image
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes))

    # Save temporarily
    temp_path = "/tmp/temp_image.jpg"
    img.save(temp_path)

    # Run inference
    result = inferencer.predict(temp_path, prompt)

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run with: python inference_api.py
# Test with: curl -X POST "http://localhost:8000/predict" \\
#              -F "image=@test.jpg" \\
#              -F "prompt=Segment the liver"
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Inference Examples")
    parser.add_argument("--example", type=str, default="basic",
                        choices=["basic", "batch", "lora", "memory", "api"],
                        help="Which example to run")
    parser.add_argument("--model", type=str, default="./exported_model",
                        help="Path to model")
    parser.add_argument("--image", type=str, default="./test_image.jpg",
                        help="Path to test image")

    args = parser.parse_args()

    examples = {
        "basic": example_basic,
        "batch": example_batch,
        "lora": example_lora,
        "memory": example_memory_efficient,
        "api": example_fastapi
    }

    examples[args.example]()
