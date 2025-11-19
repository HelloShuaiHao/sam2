"""LoRA trainer implementation using PEFT and HuggingFace Transformers."""

from pathlib import Path
from typing import Any, Dict, Optional
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training

from .base_trainer import BaseTrainer
from ..config import TrainingConfig, TrainingMethod

logger = logging.getLogger(__name__)


class LoRATrainer(BaseTrainer):
    """Trainer for LoRA fine-tuning of vision-language models.

    Uses PEFT (Parameter-Efficient Fine-Tuning) library for LoRA
    and HuggingFace Transformers for training loop.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize LoRA trainer.

        Args:
            config: Training configuration
        """
        super().__init__(config)
        self.peft_config = None
        self.trainer = None

    def setup(self) -> None:
        """Set up model, tokenizer, and LoRA configuration."""
        logger.info(f"Setting up LoRA trainer for model: {self.config.model.name}")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            cache_dir=self.config.model.cache_dir,
            trust_remote_code=True
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare quantization config for QLoRA
        quantization_config = None
        if self.config.training.method == TrainingMethod.QLORA:
            logger.info("Setting up QLoRA with 4-bit quantization...")
            quant_cfg = self.config.training.quantization

            # Determine compute dtype
            compute_dtype = torch.bfloat16 if quant_cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=quant_cfg.load_in_4bit,
                load_in_8bit=quant_cfg.load_in_8bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
            )
            logger.info(f"  - Quantization: {quant_cfg.bnb_4bit_quant_type}")
            logger.info(f"  - Compute dtype: {quant_cfg.bnb_4bit_compute_dtype}")
            logger.info(f"  - Double quantization: {quant_cfg.bnb_4bit_use_double_quant}")

        # Load base model
        logger.info("Loading base model...")
        load_kwargs = {
            "cache_dir": self.config.model.cache_dir,
            "trust_remote_code": True,
        }

        if quantization_config is not None:
            # QLoRA mode: use quantization config
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"  # Required for quantization
        else:
            # Regular LoRA mode: use mixed precision
            load_kwargs["torch_dtype"] = torch.bfloat16 if self.config.hardware.mixed_precision.value == "bf16" else torch.float16
            load_kwargs["device_map"] = self.config.hardware.device if self.config.hardware.device == "auto" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            **load_kwargs
        )

        # Prepare model for k-bit training (QLoRA specific)
        if self.config.training.method == TrainingMethod.QLORA:
            logger.info("Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.hardware.gradient_checkpointing
            )
        elif self.config.hardware.gradient_checkpointing:
            # Enable gradient checkpointing for regular LoRA
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Set up LoRA configuration
        lora_config_dict = self.config.training.lora
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config_dict.rank,
            lora_alpha=lora_config_dict.alpha,
            lora_dropout=lora_config_dict.dropout,
            bias=lora_config_dict.bias,
            target_modules=lora_config_dict.target_modules or self._get_default_target_modules()
        )

        # Apply LoRA to model
        logger.info(f"Applying LoRA with rank={lora_config_dict.rank}, alpha={lora_config_dict.alpha}")
        self.model = get_peft_model(self.model, self.peft_config)

        # Print trainable parameters
        trainable_params, all_params = self._get_trainable_params()
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )

    def train(self, train_dataset: Any, eval_dataset: Optional[Any] = None) -> Dict[str, Any]:
        """Execute LoRA training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Training results
        """
        if self.model is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")

        logger.info("Starting LoRA training...")

        # Create training arguments
        training_args = self._create_training_arguments()

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Start training
        train_result = self.trainer.train(
            resume_from_checkpoint=self.config.checkpointing.resume_from_checkpoint
        )

        # Save final model
        self.trainer.save_model(self.config.checkpointing.output_dir)

        # Log training metrics
        metrics = train_result.metrics
        logger.info(f"Training completed. Final loss: {metrics.get('train_loss', 'N/A')}")

        return {
            "metrics": metrics,
            "output_dir": self.config.checkpointing.output_dir
        }

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation.

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() first.")

        logger.info("Running evaluation...")
        eval_results = self.trainer.evaluate()
        logger.info(f"Evaluation loss: {eval_results.get('eval_loss', 'N/A')}")

        return eval_results

    def save_checkpoint(self, output_dir: Path) -> None:
        """Save LoRA checkpoint.

        Args:
            output_dir: Directory to save checkpoint
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters only (lightweight)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load PEFT model
        self.model = PeftModel.from_pretrained(
            self.model,
            checkpoint_path,
            is_trainable=True
        )

        logger.info("Checkpoint loaded successfully")

    def merge_and_save(self, output_dir: Path) -> None:
        """Merge LoRA weights with base model and save.

        Args:
            output_dir: Directory to save merged model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Merging LoRA weights with base model...")

        # Merge weights
        merged_model = self.model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Merged model saved to {output_dir}")

    def _create_training_arguments(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments from config.

        Returns:
            TrainingArguments object
        """
        return TrainingArguments(
            output_dir=self.config.checkpointing.output_dir,
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            warmup_ratio=self.config.training.warmup_ratio,
            logging_steps=self.config.logging.log_steps,
            save_steps=self.config.checkpointing.save_steps,
            save_total_limit=self.config.checkpointing.save_total_limit,
            fp16=(self.config.hardware.mixed_precision.value == "fp16"),
            bf16=(self.config.hardware.mixed_precision.value == "bf16"),
            gradient_checkpointing=self.config.hardware.gradient_checkpointing,
            dataloader_num_workers=self.config.hardware.num_workers,
            logging_dir=self.config.logging.tensorboard_dir,
            report_to=self.config.logging.report_to,
            seed=self.config.seed,
            remove_unused_columns=False,  # Important for vision-language models
        )

    def _get_default_target_modules(self) -> list[str]:
        """Get default LoRA target modules based on model type.

        Returns:
            List of module names to apply LoRA to
        """
        # Common target modules for different model architectures
        if "llama" in self.config.model.name.lower() or "vicuna" in self.config.model.name.lower():
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "qwen" in self.config.model.name.lower():
            return ["c_attn", "c_proj"]
        else:
            # Generic targets that work for most transformers
            return ["q_proj", "v_proj"]

    def _get_trainable_params(self) -> tuple[int, int]:
        """Count trainable and total parameters.

        Returns:
            Tuple of (trainable_params, all_params)
        """
        trainable_params = 0
        all_params = 0

        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return trainable_params, all_params
