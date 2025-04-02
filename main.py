# Here are the key changes needed to fix CUDA usage:

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

    # Don't move tensors to device here - do it in the training loop
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()  # For causal language modeling
    }


class MovieLLMTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        # Improved device selection
        if torch.cuda.is_available():
            # Select specific device if provided
            if hasattr(config, 'cuda_device_id') and config.cuda_device_id < torch.cuda.device_count():
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

    def setup_training(self):
        """Initialize model, tokenizer, datasets, and optimizer with improved CUDA handling"""
        logger.info("Setting up training components...")

        # Pre-allocate CUDA cache if needed
        if torch.cuda.is_available():
            # Optional: Pre-allocate memory to reduce fragmentation
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            # Benchmark mode can improve performance for fixed input sizes
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

        # Move model to appropriate device - IMPORTANT FIX
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

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
            weight_decay=0.01,  # Add weight decay for regularization
            betas=(0.9, 0.999),  # Standard beta values
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

    def save_checkpoint(self, val_loss: float, is_best: bool = False, is_emergency: bool = False) -> None:
        """Save model checkpoint with proper error handling for CUDA devices"""
        try:
            # Create a copy of the model on CPU for saving
            model_to_save = type(self.model)(self.model.config)
            model_to_save.load_state_dict(self.model.state_dict())
            model_to_save.to('cpu')

            checkpoint = {
                'epoch': self.global_step,
                'model_state_dict': model_to_save.state_dict(),  # Already on CPU
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'val_loss': val_loss,
                'config': vars(self.config),
            }

            # Save EMA state if used
            if self.ema is not None:
                ema_state = {}
                for k, v in self.ema.shadow.items():
                    ema_state[k] = v.detach().cpu()  # Move to CPU
                checkpoint['ema_state'] = ema_state

            # Determine checkpoint file path
            if is_emergency:
                checkpoint_path = self.output_dir / 'emergency_checkpoint.pt'
            else:
                checkpoint_path = self.output_dir / f'checkpoint-{self.global_step}.pt'

            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # If this is the best model so far, save it separately
            if is_best:
                best_path = self.output_dir / 'best_model.pt'
                shutil.copy(checkpoint_path, best_path)
                logger.info(f"Saved best model with val_loss={val_loss:.4f}")

            # Keep only the last 3 checkpoints to save space (skip for emergency)
            if not is_emergency:
                checkpoints = sorted(list(self.output_dir.glob('checkpoint-*.pt')))
                for checkpoint in checkpoints[:-3]:
                    checkpoint.unlink()

            # Clean up memory after saving
            del checkpoint, model_to_save
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            # Try emergency save with minimal state
            try:
                minimal_checkpoint = {
                    'global_step': self.global_step,
                    'val_loss': val_loss
                }
                emergency_path = self.output_dir / 'emergency_minimal_checkpoint.pt'
                torch.save(minimal_checkpoint, emergency_path)
                logger.info(f"Saved minimal emergency checkpoint to {emergency_path}")
            except Exception as e2:
                logger.error(f"Even minimal checkpoint saving failed: {e2}")

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
                # Move data to device - IMPORTANT FIX
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
                        # Move batch to device - IMPORTANT FIX
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

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
                            batch_adjusted = self._maybe_adjust_batch_size(oom_detected=True)

                            # If we adjusted batch size, rebuild dataloader
                            if batch_adjusted:
                                # Re-create dataloader with new batch size
                                self.train_dataloader = DataLoader(
                                    self.train_dataloader.dataset,
                                    batch_size=self.current_batch_size,
                                    shuffle=True,
                                    num_workers=self.config.num_workers,
                                    pin_memory=self.config.pin_memory and torch.cuda.is_available(),
                                    drop_last=True,
                                )

                                # Log new batch size
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


# Fix the EMA class to properly handle CUDA tensors
class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

        # Determine device from model
        self.device = next(model.parameters()).device
        logger.info(f"EMA initialized on device: {self.device}")

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


# Fix in main() to ensure CUDA is properly used
def main():
    try:
        # Set multiprocessing start method to 'spawn' for better CUDA compatibility
        import multiprocessing
        if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
            # Windows and MacOS should use 'spawn'
            multiprocessing.set_start_method('spawn', force=True)
        else:
            # On Linux, 'fork' is faster but can cause issues with CUDA
            # 'spawn' is safer but slower
            multiprocessing.set_start_method('spawn', force=True)

        # Set database path in a location with adequate storage
        db_path = os.path.expanduser("~/movie_data.sqlite")

        # Initialize config with CUDA-optimized settings
        config = TrainingConfig(
            db_path=db_path,
            batch_size=4,  # Start with small batch size
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
            cuda_device_id=0  # Use first GPU by default
        )

        # Print CUDA information before starting
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA current device: {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
                logger.info(f"  CUDA Capability: {props.major}.{props.minor}")

            # Set environment variables to optimize CUDA performance
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Larger workspace for CUBLAS
        else:
            logger.warning("CUDA is not available. Training will be slow on CPU.")

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
        raise


if __name__ == "__main__":
    main()