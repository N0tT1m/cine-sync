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
import wandb  # For tracking training progress
from tqdm import tqdm
import shutil

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


class MovieDataProcessor:
    def __init__(self):
        self.actors_data: Dict[int, Actor] = {}
        self.movies_df: Optional[pd.DataFrame] = None
        self.tv_df: Optional[pd.DataFrame] = None
        self.statistics: Dict = {}
        self.movielens_data: Dict[str, pd.DataFrame] = {}
        self.netflix_data: Dict[str, pd.DataFrame] = {}

    def load_all_data(self, movielens_path: str = './ml-32m',
                      netflix_path: str = './archive',
                      tmdb_path: str = './tmdb') -> None:
        """Load all datasets from their respective paths"""
        try:
            logger.info("Starting to load all datasets...")

            # Load MovieLens data
            self.load_movielens_data(base_path=movielens_path)
            logger.info("MovieLens data loaded successfully")

            # Load Netflix data
            self.load_netflix_data(base_path=netflix_path)
            logger.info("Netflix data loaded successfully")

            # Load TMDB data
            self.load_tmdb_data(base_path=tmdb_path)
            logger.info("TMDB data loaded successfully")

            logger.info("All datasets loaded successfully")

        except Exception as e:
            logger.error(f"Error loading all data: {e}")
            raise

    def load_movielens_data(self, base_path: str = './ml-32m') -> None:
        """Load MovieLens dataset from multiple files"""
        try:
            # Load each file
            files = {
                'links': 'links.csv',
                'movies': 'movies.csv',
                'ratings': 'ratings.csv',
                'tags': 'tags.csv'
            }

            for key, filename in files.items():
                filepath = os.path.join(base_path, filename)
                self.movielens_data[key] = pd.read_csv(filepath)
                logger.info(f"Loaded MovieLens {key} data from {filepath}")

            # Process movies data
            self.movielens_data['movies']['genres'] = self.movielens_data['movies']['genres'].str.split('|')

            # Process ratings data
            self.movielens_data['ratings'].sort_values(['userId', 'timestamp'], inplace=True)

            logger.info("Successfully loaded all MovieLens data")
        except Exception as e:
            logger.error(f"Error loading MovieLens data: {e}")
            raise

    def load_netflix_data(self, base_path: str = './archive') -> None:
        """Load Netflix challenge dataset from multiple files"""
        try:
            # Load movie titles with proper handling of commas in titles
            movie_titles_path = os.path.join(base_path, 'movie_titles.csv')

            # Custom parsing for movie titles with explicit encoding
            movies_data = []
            with open(movie_titles_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Split only on first two commas
                    parts = line.strip().split(',', 2)
                    if len(parts) == 3:
                        movie_id, year, title = parts
                        movies_data.append({
                            'movie_id': int(movie_id),
                            'year': int(year) if year != 'NULL' else None,
                            'title': title
                        })
                    else:
                        logger.warning(f"Skipping malformed line: {line.strip()}")

            self.netflix_data['movies'] = pd.DataFrame(movies_data)
            logger.info(f"Loaded {len(movies_data)} movie titles")

            # Initialize ratings dataframe
            ratings_list = []

            # Process each combined data file
            for i in range(1, 5):
                filename = f'combined_data_{i}.txt'
                filepath = os.path.join(base_path, filename)
                logger.info(f"Processing Netflix ratings file: {filename}")

                current_movie_id = None
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line.endswith(':'):
                            current_movie_id = int(line[:-1])
                        else:
                            user_id, rating, date = line.split(',')
                            ratings_list.append({
                                'movie_id': current_movie_id,
                                'user_id': int(user_id),
                                'rating': float(rating),
                                'date': pd.to_datetime(date)
                            })

                        # Process in chunks to manage memory
                        if len(ratings_list) >= 1000000:
                            chunk_df = pd.DataFrame(ratings_list)
                            if 'ratings' not in self.netflix_data:
                                self.netflix_data['ratings'] = chunk_df
                            else:
                                self.netflix_data['ratings'] = pd.concat([self.netflix_data['ratings'], chunk_df])
                            ratings_list = []

            # Process any remaining ratings
            if ratings_list:
                chunk_df = pd.DataFrame(ratings_list)
                if 'ratings' not in self.netflix_data:
                    self.netflix_data['ratings'] = chunk_df
                else:
                    self.netflix_data['ratings'] = pd.concat([self.netflix_data['ratings'], chunk_df])

            logger.info("Successfully loaded all Netflix data")

        except Exception as e:
            logger.error(f"Error loading Netflix data: {e}")
            raise

    def load_tmdb_data(self, base_path: str = './tmdb') -> None:
        """Load TMDB actor filmography data"""
        try:
            # Load movies data with explicit encoding
            movies_path = os.path.join(base_path, 'actor_filmography_data_movies.csv')
            self.movies_df = pd.read_csv(movies_path, encoding='utf-8')

            # Load TV data with explicit encoding
            tv_path = os.path.join(base_path, 'actor_filmography_data_tv.csv')
            self.tv_df = pd.read_csv(tv_path, encoding='utf-8')

            # Load JSON data with explicit encoding
            json_path = os.path.join(base_path, 'actor_filmography_data.json')
            with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)

            for actor_id, actor_data in data.items():
                actor = Actor(
                    id=int(actor_id),
                    name=actor_data['name'],
                    profile_path=actor_data['profile_path'],
                    popularity=actor_data['popularity'],
                    credits=[
                        ActorCredit(
                            media_id=credit['media_id'],
                            media_type=credit['media_type'],
                            title=credit['title'],
                            character=credit['character'],
                            release_date=credit['release_date'],
                            genres=credit['genres']
                        ) for credit in actor_data['credits']
                    ]
                )
                self.actors_data[int(actor_id)] = actor

            logger.info("Successfully loaded all TMDB data")
        except Exception as e:
            logger.error(f"Error loading TMDB data: {e}")
            raise

    def analyze_ratings_distribution(self) -> Dict:
        """Analyze ratings distribution across different datasets"""
        distributions = {}

        if self.movielens_data and 'ratings' in self.movielens_data:
            distributions['movielens'] = {
                'mean': self.movielens_data['ratings']['rating'].mean(),
                'median': self.movielens_data['ratings']['rating'].median(),
                'std': self.movielens_data['ratings']['rating'].std(),
                'count': len(self.movielens_data['ratings'])
            }

        if self.netflix_data and 'ratings' in self.netflix_data:
            distributions['netflix'] = {
                'mean': self.netflix_data['ratings']['rating'].mean(),
                'median': self.netflix_data['ratings']['rating'].median(),
                'std': self.netflix_data['ratings']['rating'].std(),
                'count': len(self.netflix_data['ratings'])
            }

        return distributions

    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics from all datasets"""
        stats = {}  # Initialize empty stats dictionary instead of calling super()

        # Add MovieLens specific statistics
        if self.movielens_data:
            stats['movielens'] = {
                'total_users': self.movielens_data['ratings'][
                    'userId'].nunique() if 'ratings' in self.movielens_data else 0,
                'total_movies': len(self.movielens_data['movies']) if 'movies' in self.movielens_data else 0,
                'total_ratings': len(self.movielens_data['ratings']) if 'ratings' in self.movielens_data else 0,
                'total_tags': len(self.movielens_data['tags']) if 'tags' in self.movielens_data else 0
            }

        # Add Netflix specific statistics
        if self.netflix_data:
            stats['netflix'] = {
                'total_movies': len(self.netflix_data['movies']) if 'movies' in self.netflix_data else 0,
                'total_ratings': len(self.netflix_data['ratings']) if 'ratings' in self.netflix_data else 0
            }

            if 'movies' in self.netflix_data and 'year' in self.netflix_data['movies'].columns:
                stats['netflix']['year_range'] = {
                    'min': int(self.netflix_data['movies']['year'].min()),
                    'max': int(self.netflix_data['movies']['year'].max())
                }

        # Add TMDB specific statistics
        if self.actors_data:
            stats['tmdb'] = {
                'total_actors': len(self.actors_data),
                'total_movies': len(self.movies_df) if self.movies_df is not None else 0,
                'total_tv_shows': len(self.tv_df) if self.tv_df is not None else 0
            }

        # Add ratings distribution
        stats['ratings_distribution'] = self.analyze_ratings_distribution()

        return stats

    def export_processed_data(self, output_file: str) -> None:
        """Export processed data to a JSON file"""
        try:
            output_data = {
                'statistics': self.calculate_statistics(),
                'timestamp': datetime.now().isoformat()
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully exported processed data to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting processed data: {e}")
            raise

    def _prepare_training_texts(self) -> List[str]:
        """Prepare text data for training from all available sources"""
        training_texts = []

        # Process MovieLens data
        if self.movielens_data and 'movies' in self.movielens_data:
            for _, movie in self.movielens_data['movies'].iterrows():
                text = f"Movie Title: {movie['title']}\nGenres: {' | '.join(movie['genres'])}\n\n"
                training_texts.append(text)

        # Process Netflix data
        if self.netflix_data and 'movies' in self.netflix_data:
            for _, movie in self.netflix_data['movies'].iterrows():
                year = str(movie['year']) if pd.notna(movie['year']) else 'Unknown'
                text = f"Movie Title: {movie['title']}\nYear: {year}\n\n"
                training_texts.append(text)

        # Process TMDB actor data
        if self.actors_data:
            for actor_id, actor in self.actors_data.items():
                credits_text = "\n".join([
                    f"Title: {credit.title}\n"
                    f"Role: {credit.character}\n"
                    f"Type: {credit.media_type}\n"
                    f"Genres: {' | '.join(credit.genres)}\n"
                    for credit in actor.credits[:5]  # Limit to top 5 credits
                ])

                text = f"Actor: {actor.name}\nPopularity: {actor.popularity}\n\nCredits:\n{credits_text}\n\n"
                training_texts.append(text)

        return training_texts

    def create_datasets(
            self,
            tokenizer: PreTrainedTokenizer,
            max_length: int = 512,
            train_size: float = 0.8,
            val_size: float = 0.1,
            test_size: float = 0.1,
            random_state: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create training, validation, and test datasets from the processed movie data.

        Args:
            tokenizer: The tokenizer to use for encoding the texts
            max_length: Maximum sequence length for the model
            train_size: Proportion of data to use for training
            val_size: Proportion of data to use for validation
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Preparing training texts...")
        all_texts = self._prepare_training_texts()

        if not all_texts:
            raise ValueError("No training texts available. Please ensure data is loaded properly.")

        logger.info(f"Created {len(all_texts)} training examples")

        # First split: separate test set
        train_val_texts, test_texts = train_test_split(
            all_texts,
            test_size=test_size,
            random_state=random_state
        )

        # Second split: separate train and validation sets
        relative_val_size = val_size / (train_size + val_size)
        train_texts, val_texts = train_test_split(
            train_val_texts,
            test_size=relative_val_size,
            random_state=random_state
        )

        logger.info(f"Split sizes - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

        # Create datasets
        train_dataset = MovieDataset(train_texts, tokenizer, max_length)
        val_dataset = MovieDataset(val_texts, tokenizer, max_length)
        test_dataset = MovieDataset(test_texts, tokenizer, max_length)

        return train_dataset, val_dataset, test_dataset

# Add new TrainingConfig dataclass
@dataclass
class TrainingConfig:
    output_dir: str = "./movie_llm_checkpoints"
    model_name: str = "gpt2"  # Base model to start from
    batch_size: int = 8
    max_length: int = 512
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_wandb: bool = True

class MovieDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        self.labels = self.encodings['input_ids'].clone()

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


class MovieLLMTrainer:
    def __init__(self, config: TrainingConfig, model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 train_dataloader: DataLoader, val_dataloader: DataLoader):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Initialize optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-8  # Add epsilon for numerical stability
        )

        # Calculate total training steps
        num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        self.max_train_steps = config.num_epochs * num_update_steps_per_epoch

        # Initialize learning rate scheduler
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=self.max_train_steps
        )

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize training tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_time = time.time()

        # Initialize wandb with retry mechanism and error handling
        if config.use_wandb:
            self._init_wandb_with_retry()

    def _init_wandb_with_retry(self, max_retries=3, retry_delay=5):
        """Initialize wandb with retry mechanism"""
        for attempt in range(max_retries):
            try:
                if wandb.run is not None:
                    wandb.finish()
                wandb.init(
                    project="movie-llm-training",
                    config=vars(self.config),
                    settings=wandb.Settings(start_method="thread")
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Failed to initialize wandb after {max_retries} attempts. "
                                   f"Continuing without wandb tracking. Error: {e}")
                    self.config.use_wandb = False
                else:
                    logger.warning(
                        f"Wandb initialization attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

    def save_checkpoint(self, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint-{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)

        # If this is the best model so far, save it separately
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            shutil.copy(checkpoint_path, best_path)

        # Keep only the last 3 checkpoints to save space
        checkpoints = sorted(self.output_dir.glob('checkpoint-*.pt'))
        for checkpoint in checkpoints[:-3]:
            checkpoint.unlink()

    def evaluate(self) -> float:
        """Evaluate the model on validation set"""
        self.model.eval()
        total_loss = 0
        total_steps = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                total_steps += 1

        avg_val_loss = total_loss / total_steps
        return avg_val_loss

    def train(self) -> None:
        """Main training loop with improved error handling and memory management"""
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Total training steps: {self.max_train_steps}")

        progress_bar = tqdm(total=self.max_train_steps, desc="Training")

        try:
            for epoch in range(self.config.num_epochs):
                self.model.train()
                epoch_loss = 0

                for step, batch in enumerate(self.train_dataloader):
                    try:
                        # Move batch to device and handle memory
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels'].to(self.device)

                            # Forward pass
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )

                            loss = outputs.loss / self.config.gradient_accumulation_steps
                            loss.backward()

                            epoch_loss += loss.item()

                        # Update weights if we've accumulated enough gradients
                        if (step + 1) % self.config.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )

                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()

                            self.global_step += 1
                            progress_bar.update(1)

                            # Log metrics with error handling
                            if self.config.use_wandb:
                                try:
                                    wandb.log({
                                        'train_loss': loss.item(),
                                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                                        'global_step': self.global_step
                                    })
                                except Exception as e:
                                    logger.warning(f"Failed to log to wandb: {e}")
                                    self.config.use_wandb = False

                            # Clean up CUDA memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Evaluate and save checkpoint if needed
                            if self.global_step % self.config.eval_steps == 0:
                                val_loss = self.evaluate()

                                if self.config.use_wandb:
                                    try:
                                        wandb.log({
                                            'val_loss': val_loss,
                                            'global_step': self.global_step
                                        })
                                    except Exception as e:
                                        logger.warning(f"Failed to log validation metrics to wandb: {e}")

                                if val_loss < self.best_val_loss:
                                    self.best_val_loss = val_loss
                                    self.save_checkpoint(val_loss, is_best=True)
                                    logger.info(f"New best validation loss: {val_loss:.4f}")

                            # Regular checkpoint saving
                            if self.global_step % self.config.save_steps == 0:
                                self.save_checkpoint(val_loss)

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning("CUDA out of memory. Attempting to recover...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                # End of epoch logging
                avg_epoch_loss = epoch_loss / len(self.train_dataloader)
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                            f"Average loss: {avg_epoch_loss:.4f}")

        except Exception as e:
            logger.error(f"Training interrupted due to error: {e}")
            # Save emergency checkpoint
            self.save_checkpoint(float('inf'), is_emergency=True)
            raise

        finally:
            # Cleanup
            if self.config.use_wandb and wandb.run is not None:
                wandb.finish()
            progress_bar.close()

def main():
    try:
        # Set multiprocessing start method to 'spawn' for better compatibility
        if sys.platform.startswith('win'):
            torch.multiprocessing.set_start_method('spawn')

        # Initialize config with reduced batch size and gradient accumulation
        config = TrainingConfig(
            batch_size=4,  # Reduced from 8
            gradient_accumulation_steps=8,  # Increased from 4
            max_length=384,  # Reduced from 512 to save memory
        )

        # Initialize processor and load data
        processor = MovieDataProcessor()
        processor.load_all_data()

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        model.resize_token_embeddings(len(tokenizer))

        # Create datasets and dataloaders with reduced num_workers
        train_dataset, val_dataset, test_dataset = processor.create_datasets(tokenizer, max_length=config.max_length)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,  # Reduced from 4
            pin_memory=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,  # Reduced from 4
            pin_memory=True
        )

        # Initialize trainer
        trainer = MovieLLMTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )

        # Start training
        trainer.train()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()