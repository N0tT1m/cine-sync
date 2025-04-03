import sys
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup, PreTrainedTokenizer
)
from pathlib import Path
import time
from datetime import datetime
import wandb
from tqdm import tqdm
import shutil
import gc
import sqlite3
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ActorCredit:
    media_id: int
    media_type: str
    title: str
    character: str
    release_date: str
    genres: List[str]


@dataclass
class Actor:
    id: int
    name: str
    profile_path: Optional[str]
    popularity: float
    credits: List[ActorCredit]


class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Determine device from model parameters
        self.device = next(model.parameters()).device
        logger.info(f"EMA initialized on device: {self.device}")

        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store on the same device as the model parameters
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # Make sure shadow is on the same device as param
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)

                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # Make sure shadow is on the same device as param
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)

                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# Database manager class remains unchanged
class DatabaseManager:
    """Class to handle all database operations"""
    # ... [previous code remains the same]


# Movie data processor class remains unchanged
class MovieDataProcessor:
    """Class to process movie data"""
    # ... [previous code remains the same]


@dataclass
class TrainingConfig:
    output_dir: str = "./movie_llm_checkpoints"
    db_path: str = "./movie_data.sqlite"
    model_name: str = "gpt2"
    batch_size: int = 4
    max_length: int = 384
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 500
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    use_wandb: bool = True

    # Improved CUDA configuration
    use_fp16: bool = True
    pin_memory: bool = True
    num_workers: int = 2
    max_examples: int = 100000
    cuda_device_id: int = 0  # Specify which CUDA device to use
    use_amp: bool = True  # Use automatic mixed precision
    dynamic_batch_size: bool = True  # Dynamically adjust batch size
    memory_efficient_fp16: bool = True  # Enable memory-efficient FP16
    ema_decay: float = 0.999  # Exponential moving average for model weights
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to save memory
    memory_monitoring: bool = True  # Enable periodic memory monitoring


class MovieDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize on-the-fly instead of pre-tokenizing everything
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()

        # Note: we don't move tensors to device here because DataLoader will handle this
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For causal language modeling
        }


class MovieLLMTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        # FIXED: Set CUDA device properly
        if torch.cuda.is_available():
            # Set the current device - this is critical!
            torch.cuda.set_device(config.cuda_device_id)

            # Select specific device if provided
            if config.cuda_device_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{config.cuda_device_id}")
            else:
                self.device = torch.device("cuda:0")  # Default to first GPU

            # Log GPU information
            device_properties = torch.cuda.get_device_properties(self.device)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"GPU Memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Capability: {device_properties.major}.{device_properties.minor}")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking variables
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # seconds

        # Set up automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and torch.cuda.is_available())

        # Track original batch size for dynamic adjustments
        self.original_batch_size = config.batch_size
        self.current_batch_size = config.batch_size

        # Initialize exponential moving average if enabled
        self.ema = None

    def _log_cuda_memory(self, context=""):
        """Log CUDA memory usage for debugging and monitoring"""
        if not torch.cuda.is_available():
            return

        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        if device.type != 'cuda':
            return

        # Get device index
        device_idx = device.index if device.index is not None else 0

        # Get memory stats
        allocated = torch.cuda.memory_allocated(device_idx) / 1024 ** 2
        reserved = torch.cuda.memory_reserved(device_idx) / 1024 ** 2
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / 1024 ** 2

        # Get device properties for total memory
        props = torch.cuda.get_device_properties(device_idx)
        total = props.total_memory / 1024 ** 2

        logger.info(f"{context}: "
                    f"Allocated: {allocated:.2f}MB | "
                    f"Reserved: {reserved:.2f}MB | "
                    f"Max Allocated: {max_allocated:.2f}MB | "
                    f"Total: {total:.2f}MB | "
                    f"Utilization: {(allocated / total) * 100:.2f}%")

        # Log to wandb if available
        if self.config.use_wandb and wandb.run is not None:
            try:
                wandb.log({
                    "cuda_allocated_mb": allocated,
                    "cuda_reserved_mb": reserved,
                    "cuda_utilization": (allocated / total) * 100,
                    "global_step": self.global_step
                })
            except Exception as e:
                logger.warning(f"Failed to log CUDA memory to wandb: {e}")

    def setup_training(self):
        """Initialize model, tokenizer, datasets, and optimizer with improved CUDA handling"""
        logger.info("Setting up training components...")

        # FIXED: Pre-allocate CUDA cache and set priority
        if torch.cuda.is_available():
            # Empty cache before initializing model
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

            # Monitor initial GPU memory
            self._log_cuda_memory("Initial GPU memory state")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model with CUDA optimizations
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Apply gradient checkpointing if enabled (saves memory)
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # FIXED: Explicitly move model to the correct device
        logger.info(f"Moving model to device: {self.device}")
        self.model = self.model.to(self.device)

        # Verify model device
        model_device = next(self.model.parameters()).device
        logger.info(f"Model is now on device: {model_device}")

        # Set up Exponential Moving Average if enabled
        if hasattr(self.config, 'ema_decay') and self.config.ema_decay > 0:
            self.ema = EMA(self.model, decay=self.config.ema_decay)
            logger.info(f"Exponential Moving Average initialized with decay={self.config.ema_decay}")

        # Set up data processor
        processor = MovieDataProcessor(db_path=self.config.db_path)

        # Create datasets with limited examples to save memory
        train_dataset, val_dataset, test_dataset = processor.create_datasets(
            self.tokenizer,
            max_length=self.config.max_length,
            max_examples=self.config.max_examples
        )

        # Create dataloaders with optimized settings for GPU
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            drop_last=True,  # Dropping last batch ensures consistent batch sizes
            persistent_workers=self.config.num_workers > 0,  # Keep workers alive between epochs
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
        )

        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        self.max_train_steps = self.config.num_epochs * num_update_steps_per_epoch

        # Initialize optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-8,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        # Initialize learning rate scheduler
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.max_train_steps
        )

        # Initialize wandb if enabled
        if self.config.use_wandb:
            self._init_wandb_with_retry()

        # Check GPU memory usage after setup
        if torch.cuda.is_available() and self.config.memory_monitoring:
            self._log_cuda_memory("GPU memory after setup")

        logger.info("Training setup complete")

    def _init_wandb_with_retry(self, max_retries=3, retry_delay=5):
        """Initialize wandb with retry mechanism"""
        # ... [remains the same]

    def save_checkpoint(self, val_loss: float, is_best: bool = False, is_emergency: bool = False) -> None:
        """Save model checkpoint with proper error handling for CUDA devices"""
        # ... [remains the same]

    def evaluate(self) -> float:
        """Evaluate the model on validation set with proper CUDA memory management"""
        self.model.eval()
        total_loss = 0
        total_steps = 0

        # Apply EMA weights for evaluation if enabled
        if self.ema is not None:
            self.ema.apply_shadow()

        # Use mixed precision for evaluation
        with torch.no_grad():
            for batch in self.val_dataloader:
                # FIXED: Move data to device - with device checking
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                try:
                    with torch.cuda.amp.autocast(enabled=self.config.use_amp and torch.cuda.is_available()):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                    total_loss += outputs.loss.item()
                    total_steps += 1

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("CUDA OOM during evaluation - skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e

                finally:
                    # Clean up CUDA memory
                    del input_ids, attention_mask, labels
                    if 'outputs' in locals():
                        del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Restore original weights if using EMA
        if self.ema is not None:
            self.ema.restore()

        # Handle case where all batches were skipped
        if total_steps == 0:
            logger.warning("No valid steps during evaluation, returning inf loss")
            return float('inf')

        avg_val_loss = total_loss / total_steps
        return avg_val_loss

    def train(self) -> None:
        """Main training loop with improved CUDA memory management"""
        # Set up all components
        self.setup_training()

        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Using mixed precision: {self.config.use_amp and torch.cuda.is_available()}")
        logger.info(f"Gradient checkpointing: {self.config.gradient_checkpointing}")

        progress_bar = tqdm(total=self.max_train_steps, desc="Training")

        try:
            for epoch in range(self.config.num_epochs):
                self.model.train()
                epoch_loss = 0
                epoch_start_time = time.time()

                for step, batch in enumerate(self.train_dataloader):
                    # Periodically check memory if enabled
                    if self.config.memory_monitoring and time.time() - self.last_memory_check > self.memory_check_interval:
                        self._log_cuda_memory(f"GPU memory at step {self.global_step}")
                        self.last_memory_check = time.time()

                    try:
                        # FIXED: Move batch to device with explicit logging on first batch
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        # For debugging - verify tensor devices on first batch
                        if step == 0 and epoch == 0:
                            logger.info(f"Input tensor device: {input_ids.device}")
                            logger.info(f"Model device: {next(self.model.parameters()).device}")

                        # Forward pass with mixed precision
                        with torch.cuda.amp.autocast(enabled=self.config.use_amp and torch.cuda.is_available()):
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )

                            loss = outputs.loss / self.config.gradient_accumulation_steps

                        # Backward pass with gradient scaling for mixed precision
                        self.scaler.scale(loss).backward()
                        epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                        # Update weights if we've accumulated enough gradients
                        if (step + 1) % self.config.gradient_accumulation_steps == 0:
                            # Clip gradients
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )

                            # Update with scaler for mixed precision
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)  # Better memory efficiency

                            # Update EMA
                            if self.ema is not None:
                                self.ema.update()

                            self.global_step += 1
                            progress_bar.update(1)

                            # Log metrics
                            if self.config.use_wandb:
                                try:
                                    wandb.log({
                                        'train_loss': loss.item() * self.config.gradient_accumulation_steps,
                                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                                        'batch_size': self.current_batch_size,
                                        'global_step': self.global_step,
                                        'epoch': epoch
                                    })
                                except Exception as e:
                                    logger.warning(f"Failed to log to wandb: {e}")
                                    self.config.use_wandb = False

                            # Evaluate and save checkpoint if needed
                            if self.global_step % self.config.eval_steps == 0:
                                # Force garbage collection before evaluation
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                val_loss = self.evaluate()
                                # Return model to training mode after evaluation
                                self.model.train()

                                logger.info(f"Step {self.global_step}: Validation Loss = {val_loss:.4f}")

                                if self.config.use_wandb:
                                    try:
                                        wandb.log({
                                            'val_loss': val_loss,
                                            'global_step': self.global_step
                                        })
                                    except Exception as e:
                                        logger.warning(f"Failed to log validation metrics to wandb: {e}")

                                # Save if best model
                                if val_loss < self.best_val_loss:
                                    self.best_val_loss = val_loss
                                    self.save_checkpoint(val_loss, is_best=True)
                                    logger.info(f"New best validation loss: {val_loss:.4f}")

                            # Regular checkpoint saving
                            if self.global_step % self.config.save_steps == 0:
                                # Get validation loss if we don't have a recent one
                                if self.global_step % self.config.eval_steps != 0:
                                    val_loss = self.evaluate()
                                    self.model.train()  # Return to training mode
                                self.save_checkpoint(val_loss)

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"CUDA out of memory at step {step}. Attempting to recover...")

                            # Clear any tensors from memory
                            if 'input_ids' in locals():
                                del input_ids
                            if 'attention_mask' in locals():
                                del attention_mask
                            if 'labels' in locals():
                                del labels
                            if 'outputs' in locals():
                                del outputs
                            if 'loss' in locals():
                                del loss

                            # Empty CUDA cache
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Try adjusting batch size
                            if self.config.dynamic_batch_size:
                                new_batch_size = max(1, self.current_batch_size // 2)
                                logger.warning(
                                    f"Reducing batch size from {self.current_batch_size} to {new_batch_size}")
                                self.current_batch_size = new_batch_size

                                # Re-create dataloader with new batch size
                                self.train_dataloader = DataLoader(
                                    self.train_dataloader.dataset,
                                    batch_size=self.current_batch_size,
                                    shuffle=True,
                                    num_workers=self.config.num_workers,
                                    pin_memory=self.config.pin_memory and torch.cuda.is_available(),
                                    drop_last=True,
                                )

                                logger.info(f"Recreated dataloader with batch size {self.current_batch_size}")
                            continue
                        else:
                            raise e

                    finally:
                        # Make sure to clean up any CUDA memory at the end of each step
                        if 'input_ids' in locals():
                            del input_ids
                        if 'attention_mask' in locals():
                            del attention_mask
                        if 'labels' in locals():
                            del labels
                        if 'outputs' in locals():
                            del outputs
                        if 'loss' in locals():
                            del loss

                # End of epoch logging
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / len(self.train_dataloader)
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                            f"Average loss: {avg_epoch_loss:.4f} - "
                            f"Time: {epoch_time:.2f}s")

                # Save checkpoint at end of each epoch
                val_loss = self.evaluate()
                self.model.train()  # Return to training mode
                self.save_checkpoint(val_loss, is_best=(val_loss < self.best_val_loss))

                # Log epoch stats to wandb
                if self.config.use_wandb:
                    try:
                        wandb.log({
                            'epoch': epoch + 1,
                            'epoch_loss': avg_epoch_loss,
                            'epoch_time': epoch_time,
                            'epoch_val_loss': val_loss
                        })
                    except Exception as e:
                        logger.warning(f"Failed to log epoch stats to wandb: {e}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save emergency checkpoint
            self.save_checkpoint(float('inf'), is_emergency=True)

        except Exception as e:
            logger.error(f"Training interrupted due to error: {e}")
            # Save emergency checkpoint
            self.save_checkpoint(float('inf'), is_emergency=True)
            raise

        finally:
            # Final cleanup
            if self.config.use_wandb and wandb.run is not None:
                wandb.finish()
            progress_bar.close()

            # Final memory report
            if torch.cuda.is_available() and self.config.memory_monitoring:
                self._log_cuda_memory("Final GPU memory state")

            logger.info("Training completed or interrupted")


def main():
    try:
        # FIXED: Print CUDA availability and details before any other code
        logger.info("==== CUDA Configuration ====")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")

            # Check and log for all available CUDA devices
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
                logger.info(f"  CUDA Capability: {props.major}.{props.minor}")

            # Set environment variables for CUDA
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
            logger.info(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        else:
            logger.warning("CUDA is not available. Training will be on CPU only!")
        logger.info("===========================")

        # Set multiprocessing start method to 'spawn' for better CUDA compatibility
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

        # Set database path in a location with adequate storage
        db_path = os.path.expanduser("~/movie_data.sqlite")

        # FIXED: Initialize config explicitly setting cuda_device_id
        config = TrainingConfig(
            db_path=db_path,
            batch_size=4,
            gradient_accumulation_steps=8,
            max_length=384,
            use_fp16=True,
            use_amp=True,
            max_examples=100000,
            num_workers=2,
            output_dir="./movie_llm_checkpoints",
            dynamic_batch_size=True,
            gradient_checkpointing=True,
            memory_monitoring=True,
            cuda_device_id=0  # Explicitly set to use first GPU
        )

        # Initialize processor and load data
        processor = MovieDataProcessor(db_path=db_path)
        processor.load_all_data()

        # Initialize trainer
        trainer = MovieLLMTrainer(config=config)

        # Start training
        trainer.train()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        # Log CUDA memory state when error occurs
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 2
            reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
            logger.error(f"CUDA memory at error: Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")
        raise


if __name__ == "__main__":
    main()