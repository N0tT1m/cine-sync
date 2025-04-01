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
import gc  # Add garbage collector for explicit memory management
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
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class DatabaseManager:
    """Class to handle all database operations"""

    def __init__(self, db_path=None):
        """Initialize the database connection"""
        # Create a temp file if no path is provided
        if db_path is None:
            # Create a temporary directory that will persist
            temp_dir = tempfile.mkdtemp(prefix="movie_data_")
            db_path = os.path.join(temp_dir, "movie_data.sqlite")

        self.db_path = db_path
        logger.info(f"Using database at: {self.db_path}")

        # Create or connect to the database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Enable foreign keys support
        self.cursor.execute("PRAGMA foreign_keys = ON")
        # Set journal mode to WAL for better performance
        self.cursor.execute("PRAGMA journal_mode = WAL")
        # Increase cache size (in KB)
        self.cursor.execute("PRAGMA cache_size = -50000")  # Approx 50MB cache

        # Create tables
        self._create_tables()

    def _create_tables(self):
        """Create the necessary database tables if they don't exist"""
        # MovieLens tables
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS movielens_movies (
            movieId INTEGER PRIMARY KEY,
            title TEXT,
            genres TEXT
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS movielens_ratings (
            userId INTEGER,
            movieId INTEGER,
            rating REAL,
            timestamp INTEGER,
            PRIMARY KEY (userId, movieId)
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS movielens_tags (
            userId INTEGER,
            movieId INTEGER,
            tag TEXT,
            timestamp INTEGER,
            PRIMARY KEY (userId, movieId, tag)
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS movielens_links (
            movieId INTEGER PRIMARY KEY,
            imdbId INTEGER,
            tmdbId INTEGER
        )
        ''')

        # Netflix tables
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS netflix_movies (
            movie_id INTEGER PRIMARY KEY,
            year INTEGER,
            title TEXT
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS netflix_ratings (
            movie_id INTEGER,
            user_id INTEGER,
            rating REAL,
            date TEXT,
            PRIMARY KEY (movie_id, user_id)
        )
        ''')

        # TMDB tables
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tmdb_actors (
            id INTEGER PRIMARY KEY,
            name TEXT,
            profile_path TEXT,
            popularity REAL
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tmdb_credits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_id INTEGER,
            media_id INTEGER,
            media_type TEXT,
            title TEXT,
            character TEXT,
            release_date TEXT,
            FOREIGN KEY (actor_id) REFERENCES tmdb_actors(id)
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tmdb_genres (
            credit_id INTEGER,
            genre TEXT,
            FOREIGN KEY (credit_id) REFERENCES tmdb_credits(id)
        )
        ''')

        # Movies and TV Shows from TMDB
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tmdb_movies (
            id INTEGER PRIMARY KEY,
            title TEXT,
            release_date TEXT,
            popularity REAL,
            vote_average REAL,
            vote_count INTEGER,
            overview TEXT
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tmdb_tv (
            id INTEGER PRIMARY KEY,
            name TEXT,
            first_air_date TEXT,
            popularity REAL,
            vote_average REAL,
            vote_count INTEGER,
            overview TEXT
        )
        ''')

        # Storage for statistics
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            category TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (category, key)
        )
        ''')

        # Create some useful indices
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_ratings_userid ON movielens_ratings(userId)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_ratings_movieid ON movielens_ratings(movieId)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_nf_ratings_userid ON netflix_ratings(user_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_nf_ratings_movieid ON netflix_ratings(movie_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_tmdb_credits_actorid ON tmdb_credits(actor_id)")

        # Commit changes
        self.conn.commit()

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

    def check_row_exists(self, table, conditions):
        """
        Check if a row exists in a table based on specified conditions

        Args:
            table (str): Table name
            conditions (dict): Dictionary mapping column names to values for the WHERE clause

        Returns:
            bool: True if row exists, False otherwise
        """
        where_clause = " AND ".join([f"{col} = ?" for col in conditions.keys()])
        query = f"SELECT 1 FROM {table} WHERE {where_clause} LIMIT 1"

        self.cursor.execute(query, tuple(conditions.values()))
        return self.cursor.fetchone() is not None

    def insert_many_with_progress(self, table, columns, values, batch_size=10000, check_duplicates=True):
        """
        Insert multiple rows with progress tracking and duplicate checking

        Args:
            table (str): Table name
            columns (list): List of column names
            values (list): List of tuples containing values to insert
            batch_size (int): Size of batches for insertion
            check_duplicates (bool): Whether to check for duplicates before insertion
        """
        total_rows = len(values)
        inserted_count = 0
        skipped_count = 0

        # Get primary key columns for this table
        self.cursor.execute(f"PRAGMA table_info({table})")
        table_info = self.cursor.fetchall()
        pk_columns = [row[1] for row in table_info if row[5] > 0]  # column name where pk flag > 0

        # If no primary keys found or duplicate checking disabled, use INSERT OR REPLACE
        if not pk_columns or not check_duplicates:
            placeholders = ",".join(["?"] * len(columns))
            columns_str = ",".join(columns)
            query = f"INSERT OR REPLACE INTO {table} ({columns_str}) VALUES ({placeholders})"

            # Insert in batches
            for i in tqdm(range(0, total_rows, batch_size), desc=f"Inserting into {table}"):
                batch = values[i:i + batch_size]
                self.cursor.executemany(query, batch)
                self.conn.commit()
                inserted_count += len(batch)

        else:
            # Use duplicate checking for tables with primary keys
            placeholders = ",".join(["?"] * len(columns))
            columns_str = ",".join(columns)
            insert_query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

            # Get indices of primary key columns in the values tuple
            pk_indices = [columns.index(col) for col in pk_columns if col in columns]

            # Insert in batches with duplicate checking
            for i in tqdm(range(0, total_rows, batch_size), desc=f"Processing {table}"):
                batch = values[i:i + batch_size]
                rows_to_insert = []

                for row in batch:
                    # Create conditions for checking if row exists
                    conditions = {pk_columns[idx]: row[pk_indices[idx]] for idx in range(len(pk_indices))}

                    if not self.check_row_exists(table, conditions):
                        rows_to_insert.append(row)
                    else:
                        skipped_count += 1

                if rows_to_insert:
                    self.cursor.executemany(insert_query, rows_to_insert)
                    self.conn.commit()
                    inserted_count += len(rows_to_insert)

        logger.info(f"Table {table}: {inserted_count} rows inserted, {skipped_count} duplicates skipped")
        return inserted_count, skipped_count

    def get_statistics(self, category):
        """Retrieve statistics for a category"""
        self.cursor.execute("SELECT key, value FROM statistics WHERE category = ?", (category,))
        results = self.cursor.fetchall()
        return {key: value for key, value in results}

    def store_statistic(self, category, key, value):
        """Store a statistic value"""
        value_str = str(value)
        if isinstance(value, dict):
            value_str = json.dumps(value)

        self.cursor.execute(
            "INSERT OR REPLACE INTO statistics (category, key, value) VALUES (?, ?, ?)",
            (category, key, value_str)
        )
        self.conn.commit()

    def execute_query(self, query, params=None):
        """Execute a SQL query and return results"""
        if params is None:
            self.cursor.execute(query)
        else:
            self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def get_random_rows(self, table, limit=1000):
        """Get a random sample of rows from a table"""
        # SQLite doesn't have a true RANDOM() function for large datasets, so this is approximate
        self.cursor.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT ?", (limit,))
        return self.cursor.fetchall()

    def create_training_texts(self, limit=None):
        """Generate training texts from database records"""
        texts = []

        # Get movie texts
        movie_limit = "" if limit is None else f" LIMIT {limit // 3}"
        self.cursor.execute(f"SELECT title, genres FROM movielens_movies{movie_limit}")
        for title, genres in self.cursor.fetchall():
            genre_list = genres.split('|') if genres else []
            text = f"Movie Title: {title}\nGenres: {' | '.join(genre_list)}\n\n"
            texts.append(text)

        # Get Netflix movie texts
        netflix_limit = "" if limit is None else f" LIMIT {limit // 3}"
        self.cursor.execute(f"SELECT title, year FROM netflix_movies{netflix_limit}")
        for title, year in self.cursor.fetchall():
            year_str = str(year) if year is not None else 'Unknown'
            text = f"Movie Title: {title}\nYear: {year_str}\n\n"
            texts.append(text)

        # Get actor texts with their credits
        actor_limit = "" if limit is None else f" LIMIT {limit // 3}"
        self.cursor.execute(f"SELECT id, name, popularity FROM tmdb_actors{actor_limit}")

        for actor_id, name, popularity in self.cursor.fetchall():
            # Get up to 5 credits for this actor
            self.cursor.execute(
                "SELECT c.title, c.character, c.media_type, c.id "
                "FROM tmdb_credits c "
                "WHERE c.actor_id = ? "
                "LIMIT 5",
                (actor_id,)
            )

            credits_data = []
            for title, character, media_type, credit_id in self.cursor.fetchall():
                # Get genres for this credit
                self.cursor.execute(
                    "SELECT genre FROM tmdb_genres WHERE credit_id = ?",
                    (credit_id,)
                )
                genres = [row[0] for row in self.cursor.fetchall()]

                credit_text = (
                    f"Title: {title}\n"
                    f"Role: {character}\n"
                    f"Type: {media_type}\n"
                    f"Genres: {' | '.join(genres)}\n"
                )
                credits_data.append(credit_text)

            if credits_data:
                text = (
                    f"Actor: {name}\n"
                    f"Popularity: {popularity}\n\n"
                    f"Credits:\n{''.join(credits_data)}\n\n"
                )
                texts.append(text)

        return texts

class MovieDataProcessor:
    def __init__(self, db_path=None):
        self.db = DatabaseManager(db_path)
        self.statistics = {}

    def __del__(self):
        """Ensure database connection is closed when the object is destroyed"""
        if hasattr(self, 'db'):
            self.db.close()

    def load_all_data(self, movielens_path: str = './ml-32m',
                      netflix_path: str = './archive',
                      tmdb_path: str = './tmdb') -> None:
        """Load all datasets from their respective paths"""
        try:
            logger.info("Starting to load all datasets...")

            # Load MovieLens data
            self.load_movielens_data(base_path=movielens_path)
            logger.info("MovieLens data loaded successfully")
            gc.collect()  # Force garbage collection

            # Load Netflix data
            self.load_netflix_data(base_path=netflix_path)
            logger.info("Netflix data loaded successfully")
            gc.collect()  # Force garbage collection

            # Load TMDB data
            self.load_tmdb_data(base_path=tmdb_path)
            logger.info("TMDB data loaded successfully")
            gc.collect()  # Force garbage collection

            logger.info("All datasets loaded successfully")

        except Exception as e:
            logger.error(f"Error loading all data: {e}")
            raise

    def load_movielens_data(self, base_path: str = './ml-32m') -> None:
        """Load MovieLens dataset from multiple files into database"""
        try:
            # Process smaller files first
            files = {
                'links': ('links.csv', ['movieId', 'imdbId', 'tmdbId']),
                'movies': ('movies.csv', ['movieId', 'title', 'genres']),
                'tags': ('tags.csv', ['userId', 'movieId', 'tag', 'timestamp'])
            }

            for key, (filename, columns) in files.items():
                filepath = os.path.join(base_path, filename)
                logger.info(f"Loading {filepath}")

                # Read and process in chunks
                chunk_size = 100000  # Adjust based on file size and memory
                for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                    # Convert to list of tuples for SQLite insertion
                    values = list(chunk[columns].itertuples(index=False, name=None))

                    # Insert into appropriate table
                    table_name = f"movielens_{key}"
                    self.db.insert_many_with_progress(table_name, columns, values)

                    # Clean up memory
                    del chunk, values
                    gc.collect()

            # Handle ratings file separately (it's typically much larger)
            ratings_file = os.path.join(base_path, 'ratings.csv')
            ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']

            logger.info(f"Loading ratings from {ratings_file}")
            for chunk in tqdm(pd.read_csv(ratings_file, chunksize=250000), desc="Loading ratings chunks"):
                values = list(chunk[ratings_columns].itertuples(index=False, name=None))
                self.db.insert_many_with_progress("movielens_ratings", ratings_columns, values)
                del chunk, values
                gc.collect()

            # Create statistics
            self._calculate_movielens_statistics()

            logger.info("Successfully loaded all MovieLens data")

        except Exception as e:
            logger.error(f"Error loading MovieLens data: {e}")
            raise

    def _calculate_movielens_statistics(self):
        """Calculate and store MovieLens statistics"""
        # Get counts
        total_users = self.db.execute_query("SELECT COUNT(DISTINCT userId) FROM movielens_ratings")[0][0]
        total_movies = self.db.execute_query("SELECT COUNT(*) FROM movielens_movies")[0][0]
        total_ratings = self.db.execute_query("SELECT COUNT(*) FROM movielens_ratings")[0][0]
        total_tags = self.db.execute_query("SELECT COUNT(*) FROM movielens_tags")[0][0]

        # Store statistics
        self.db.store_statistic('movielens', 'total_users', total_users)
        self.db.store_statistic('movielens', 'total_movies', total_movies)
        self.db.store_statistic('movielens', 'total_ratings', total_ratings)
        self.db.store_statistic('movielens', 'total_tags', total_tags)

        # Get rating statistics - using random sampling for large datasets
        self.db.cursor.execute(
            "SELECT AVG(rating), MIN(rating), MAX(rating) FROM ("
            "SELECT rating FROM movielens_ratings ORDER BY RANDOM() LIMIT 1000000)"
        )
        avg_rating, min_rating, max_rating = self.db.cursor.fetchone()

        self.db.store_statistic('movielens', 'avg_rating', avg_rating)
        self.db.store_statistic('movielens', 'min_rating', min_rating)
        self.db.store_statistic('movielens', 'max_rating', max_rating)

    def load_netflix_data(self, base_path: str = './archive') -> None:
        """Load Netflix challenge dataset into database with extra error handling"""
        try:
            # Check if the movie_titles.csv file exists in the expected location
            movie_titles_path = os.path.join(base_path, 'movie_titles.csv')
            if not os.path.exists(movie_titles_path):
                # Try alternative locations
                alternative_paths = [
                    os.path.join(base_path, 'netflix-prize-data', 'movie_titles.csv'),
                    os.path.join(base_path, 'data', 'movie_titles.csv')
                ]

                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        movie_titles_path = alt_path
                        logger.info(f"Found movie_titles.csv at alternative path: {alt_path}")
                        break
                else:
                    logger.error(f"Could not find movie_titles.csv in {base_path} or subdirectories")
                    raise FileNotFoundError(f"movie_titles.csv not found in {base_path}")

            logger.info(f"Loading Netflix movie titles from {movie_titles_path}")

            # Process movie titles
            movies_data = []
            with open(movie_titles_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Handle different possible formats of the movie_titles.csv file
                    try:
                        # Try comma-based parsing first
                        parts = line.strip().split(',', 2)
                        if len(parts) >= 2:  # Some entries might not have titles
                            movie_id = parts[0]
                            year = parts[1] if len(parts) > 1 else 'NULL'
                            title = parts[2] if len(parts) > 2 else ''

                            movie_id = int(movie_id)
                            year = int(year) if year != 'NULL' and year.isdigit() else None
                            movies_data.append((movie_id, year, title))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping malformed line: {line.strip()} - Error: {e}")

            if not movies_data:
                logger.warning("No movie data was successfully parsed from movie_titles.csv")
            else:
                logger.info(f"Successfully parsed {len(movies_data)} movies")

                # Insert movie data
                self.db.insert_many_with_progress('netflix_movies', ['movie_id', 'year', 'title'], movies_data)

            del movies_data
            gc.collect()

            # Find the combined data files
            combined_data_files = []

            # Check direct path
            for i in range(1, 5):
                filename = f'combined_data_{i}.txt'
                filepath = os.path.join(base_path, filename)
                if os.path.exists(filepath):
                    combined_data_files.append(filepath)

            # If not found, check subdirectories
            if not combined_data_files:
                for subdir in ['netflix-prize-data', 'data', 'training_set']:
                    for i in range(1, 5):
                        filename = f'combined_data_{i}.txt'
                        filepath = os.path.join(base_path, subdir, filename)
                        if os.path.exists(filepath):
                            combined_data_files.append(filepath)
                            break

                    # Also check numbered directories (training_set/1, training_set/2, etc.)
                    for i in range(1, 5):
                        subdir_path = os.path.join(base_path, subdir, str(i))
                        if os.path.exists(subdir_path):
                            for file in os.listdir(subdir_path):
                                if file.endswith('.txt'):
                                    filepath = os.path.join(subdir_path, file)
                                    combined_data_files.append(filepath)

            if not combined_data_files:
                logger.error("Could not find any combined_data_*.txt files")
                raise FileNotFoundError("No combined_data_*.txt files found")

            logger.info(f"Found {len(combined_data_files)} data files: {combined_data_files}")

            # Process each data file
            for filepath in combined_data_files:
                filename = os.path.basename(filepath)
                logger.info(f"Processing Netflix ratings file: {filepath}")

                current_movie_id = None
                ratings_batch = []
                batch_size = 500000  # Adjust based on available memory
                line_count = 0

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(tqdm(f, desc=f"Processing {filename}")):
                            line = line.strip()
                            line_count += 1

                            # Skip empty lines
                            if not line:
                                continue

                            # Check if this is a movie ID line
                            if ':' in line:
                                try:
                                    # Extract movie ID - handle both "movieId:" and "movieId," formats
                                    movie_part = line.split(':')[0].strip()
                                    current_movie_id = int(movie_part)
                                except (ValueError, IndexError):
                                    logger.warning(f"Invalid movie ID line: {line}")
                                    continue
                            elif current_movie_id is not None:
                                # This should be a rating line
                                try:
                                    parts = line.split(',')
                                    if len(parts) >= 3:
                                        user_id, rating, date = parts[0], parts[1], parts[2]
                                        ratings_batch.append((
                                            current_movie_id,
                                            int(user_id),
                                            float(rating),
                                            date
                                        ))
                                except (ValueError, IndexError) as e:
                                    if line_count < 10:  # Only log errors for the first few lines to avoid flooding
                                        logger.warning(f"Error parsing rating line: {line} - {e}")
                                    continue

                            # Process in batches to manage memory
                            if len(ratings_batch) >= batch_size:
                                try:
                                    self.db.insert_many_with_progress(
                                        'netflix_ratings',
                                        ['movie_id', 'user_id', 'rating', 'date'],
                                        ratings_batch
                                    )
                                except Exception as e:
                                    logger.error(f"Error inserting batch into database: {e}")
                                    # Continue with empty batch instead of failing completely

                                ratings_batch = []
                                gc.collect()  # Force garbage collection

                    # Process any remaining ratings
                    if ratings_batch:
                        try:
                            self.db.insert_many_with_progress(
                                'netflix_ratings',
                                ['movie_id', 'user_id', 'rating', 'date'],
                                ratings_batch
                            )
                        except Exception as e:
                            logger.error(f"Error inserting final batch into database: {e}")

                    del ratings_batch
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error processing file {filepath}: {e}")
                    # Continue with next file instead of failing completely

            # Calculate Netflix statistics if we have any data
            try:
                self._calculate_netflix_statistics()
                logger.info("Successfully calculated Netflix statistics")
            except Exception as e:
                logger.error(f"Error calculating Netflix statistics: {e}")

            logger.info("Completed Netflix data loading process")

        except Exception as e:
            logger.error(f"Error loading Netflix data: {e}")
            # Don't raise the exception, just log it and continue
            # This allows the program to continue with other datasets even if Netflix fails

    def _calculate_netflix_statistics(self):
        """Calculate and store Netflix statistics with extra error handling"""
        try:
            # Check if tables exist and have data
            movie_count = self.db.execute_query("SELECT COUNT(*) FROM netflix_movies")[0][0]
            if movie_count == 0:
                logger.warning("No Netflix movies in database, skipping statistics calculation")
                self.db.store_statistic('netflix', 'total_movies', 0)
                self.db.store_statistic('netflix', 'data_status', "No movie data available")
                return

            # Get counts with error handling
            try:
                total_movies = movie_count
                self.db.store_statistic('netflix', 'total_movies', total_movies)
            except Exception as e:
                logger.error(f"Error calculating movie count: {e}")
                self.db.store_statistic('netflix', 'total_movies', -1)

            try:
                total_ratings = self.db.execute_query("SELECT COUNT(*) FROM netflix_ratings")[0][0]
                self.db.store_statistic('netflix', 'total_ratings', total_ratings)
            except Exception as e:
                logger.error(f"Error calculating rating count: {e}")
                self.db.store_statistic('netflix', 'total_ratings', -1)

            try:
                total_users = self.db.execute_query("SELECT COUNT(DISTINCT user_id) FROM netflix_ratings")[0][0]
                self.db.store_statistic('netflix', 'total_users', total_users)
            except Exception as e:
                logger.error(f"Error calculating user count: {e}")
                self.db.store_statistic('netflix', 'total_users', -1)

            # Get year range with error handling
            try:
                self.db.cursor.execute("SELECT MIN(year), MAX(year) FROM netflix_movies WHERE year IS NOT NULL")
                result = self.db.cursor.fetchone()
                if result and None not in result:
                    min_year, max_year = result
                    self.db.store_statistic('netflix', 'min_year', min_year)
                    self.db.store_statistic('netflix', 'max_year', max_year)
            except Exception as e:
                logger.error(f"Error calculating year range: {e}")

            # Get rating statistics with error handling
            try:
                # Check if we have any ratings
                rating_count = self.db.execute_query("SELECT COUNT(*) FROM netflix_ratings")[0][0]
                if rating_count > 0:
                    # Sample size should be smaller than the total number of ratings
                    sample_size = min(1000000, rating_count)

                    self.db.cursor.execute(
                        f"SELECT AVG(rating), MIN(rating), MAX(rating) FROM ("
                        f"SELECT rating FROM netflix_ratings ORDER BY RANDOM() LIMIT {sample_size})"
                    )
                    result = self.db.cursor.fetchone()
                    if result and None not in result:
                        avg_rating, min_rating, max_rating = result
                        self.db.store_statistic('netflix', 'avg_rating', avg_rating)
                        self.db.store_statistic('netflix', 'min_rating', min_rating)
                        self.db.store_statistic('netflix', 'max_rating', max_rating)
            except Exception as e:
                logger.error(f"Error calculating rating statistics: {e}")

        except Exception as e:
            logger.error(f"Error in Netflix statistics calculation: {e}")
            # Store minimal statistics to avoid errors later
            self.db.store_statistic('netflix', 'data_status', "Statistics calculation failed")

    def load_tmdb_data(self, base_path: str = './tmdb') -> None:
        """Load TMDB actor filmography data into database"""
        try:
            # Load movies data
            movies_path = os.path.join(base_path, 'actor_filmography_data_movies.csv')
            logger.info(f"Loading TMDB movies from {movies_path}")

            if os.path.exists(movies_path):
                for chunk in pd.read_csv(movies_path, encoding='utf-8', chunksize=50000):
                    # Check available columns first and map to expected columns
                    available_columns = chunk.columns.tolist()
                    expected_columns = ['id', 'title', 'release_date', 'popularity', 'vote_average', 'vote_count',
                                        'overview']

                    # Log column mismatch for debugging
                    if not all(col in available_columns for col in expected_columns):
                        logger.warning(
                            f"Column mismatch in TMDB movies: Expected {expected_columns}, got {available_columns}")
                        # Only use columns that exist in the file
                        columns_to_use = [col for col in expected_columns if col in available_columns]
                        # If id column is missing, we can't proceed with this file
                        if 'id' not in columns_to_use:
                            logger.error(f"Required column 'id' missing in TMDB movies CSV, skipping file")
                            break
                    else:
                        columns_to_use = expected_columns

                    # Extract only available columns
                    values = []
                    for _, row in chunk.iterrows():
                        # Create a tuple with all expected columns, using None for missing ones
                        row_values = []
                        for col in expected_columns:
                            row_values.append(row.get(col) if col in available_columns else None)
                        values.append(tuple(row_values))

                    self.db.insert_many_with_progress('tmdb_movies', expected_columns, values)
                    del chunk, values
                    gc.collect()
            else:
                logger.warning(f"TMDB movies file not found at {movies_path}")

            # Load TV data
            tv_path = os.path.join(base_path, 'actor_filmography_data_tv.csv')
            logger.info(f"Loading TMDB TV shows from {tv_path}")

            if os.path.exists(tv_path):
                for chunk in pd.read_csv(tv_path, encoding='utf-8', chunksize=50000):
                    # Check available columns first and map to expected columns
                    available_columns = chunk.columns.tolist()
                    expected_columns = ['id', 'name', 'first_air_date', 'popularity', 'vote_average', 'vote_count',
                                        'overview']

                    # Log column mismatch for debugging
                    if not all(col in available_columns for col in expected_columns):
                        logger.warning(
                            f"Column mismatch in TMDB TV: Expected {expected_columns}, got {available_columns}")
                        # Only use columns that exist in the file
                        columns_to_use = [col for col in expected_columns if col in available_columns]
                        # If id column is missing, we can't proceed with this file
                        if 'id' not in columns_to_use:
                            logger.error(f"Required column 'id' missing in TMDB TV CSV, skipping file")
                            break
                    else:
                        columns_to_use = expected_columns

                    # Extract only available columns
                    values = []
                    for _, row in chunk.iterrows():
                        # Create a tuple with all expected columns, using None for missing ones
                        row_values = []
                        for col in expected_columns:
                            row_values.append(row.get(col) if col in available_columns else None)
                        values.append(tuple(row_values))

                    self.db.insert_many_with_progress('tmdb_tv', expected_columns, values)
                    del chunk, values
                    gc.collect()
            else:
                logger.warning(f"TMDB TV file not found at {tv_path}")

            # Load JSON data
            json_path = os.path.join(base_path, 'actor_filmography_data.json')
            logger.info(f"Loading TMDB actor data from {json_path}")

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)

                # Process actors and their credits
                actor_count = 0
                actor_batch = []
                credit_batch = []
                genre_batch = []

                for actor_id, actor_data in tqdm(data.items(), desc="Processing actors"):
                    try:
                        actor_id = int(actor_id)

                        # Check if required fields are present
                        if not all(k in actor_data for k in ['name', 'profile_path', 'popularity', 'credits']):
                            logger.warning(f"Skipping actor {actor_id} due to missing required fields")
                            continue

                        actor_batch.append((
                            actor_id,
                            actor_data['name'],
                            actor_data['profile_path'],
                            actor_data['popularity']
                        ))

                        # Process credits for this actor
                        for credit in actor_data['credits']:
                            # Check if required fields are present in the credit
                            if not all(k in credit for k in
                                       ['media_id', 'media_type', 'title', 'character', 'release_date', 'genres']):
                                continue

                            # Insert credit
                            credit_id = len(credit_batch) + 1  # Use a temporary ID
                            credit_batch.append((
                                credit_id,
                                actor_id,
                                credit['media_id'],
                                credit['media_type'],
                                credit['title'],
                                credit['character'],
                                credit['release_date']
                            ))

                            # Process genres for this credit
                            for genre in credit['genres']:
                                genre_batch.append((credit_id, genre))

                        actor_count += 1

                        # Insert in batches
                        if actor_count % 1000 == 0:
                            self.db.insert_many_with_progress('tmdb_actors',
                                                              ['id', 'name', 'profile_path', 'popularity'],
                                                              actor_batch)
                            actor_batch = []

                            self.db.insert_many_with_progress('tmdb_credits',
                                                              ['id', 'actor_id', 'media_id', 'media_type', 'title',
                                                               'character', 'release_date'],
                                                              credit_batch)
                            credit_batch = []

                            self.db.insert_many_with_progress('tmdb_genres',
                                                              ['credit_id', 'genre'],
                                                              genre_batch)
                            genre_batch = []

                            # Force garbage collection
                            gc.collect()

                    except Exception as e:
                        logger.warning(f"Error processing actor {actor_id}: {e}")

                # Insert any remaining data
                if actor_batch:
                    self.db.insert_many_with_progress('tmdb_actors',
                                                      ['id', 'name', 'profile_path', 'popularity'],
                                                      actor_batch)

                if credit_batch:
                    self.db.insert_many_with_progress('tmdb_credits',
                                                      ['id', 'actor_id', 'media_id', 'media_type', 'title', 'character',
                                                       'release_date'],
                                                      credit_batch)

                if genre_batch:
                    self.db.insert_many_with_progress('tmdb_genres',
                                                      ['credit_id', 'genre'],
                                                      genre_batch)
            else:
                logger.warning(f"TMDB actor data file not found at {json_path}")

            # Calculate TMDB statistics even if some files are missing
            self._calculate_tmdb_statistics()

            logger.info("TMDB data loading completed")

        except Exception as e:
            logger.error(f"Error loading TMDB data: {e}")
            # Store error in statistics for tracking
            self.db.store_statistic('tmdb', 'load_error', str(e))
            # Don't raise exception to allow the program to continue with other datasets
            logger.info("Continuing despite TMDB data loading error")

    def _calculate_tmdb_statistics(self):
        """Calculate and store TMDB statistics with error handling"""
        try:
            # Check if tables exist and have data
            actor_count = self.db.execute_query("SELECT COUNT(*) FROM tmdb_actors")[0][0]
            movie_count = self.db.execute_query("SELECT COUNT(*) FROM tmdb_movies")[0][0]
            tv_count = self.db.execute_query("SELECT COUNT(*) FROM tmdb_tv")[0][0]

            # Store basic counts
            self.db.store_statistic('tmdb', 'total_actors', actor_count)
            self.db.store_statistic('tmdb', 'total_movies', movie_count)
            self.db.store_statistic('tmdb', 'total_tv_shows', tv_count)

            # Get credit count if available
            try:
                credit_count = self.db.execute_query("SELECT COUNT(*) FROM tmdb_credits")[0][0]
                self.db.store_statistic('tmdb', 'total_credits', credit_count)
            except:
                self.db.store_statistic('tmdb', 'total_credits', 0)

            # Get most popular actors (top 10) if available
            try:
                if actor_count > 0:
                    self.db.cursor.execute(
                        "SELECT name, popularity FROM tmdb_actors ORDER BY popularity DESC LIMIT 10"
                    )
                    top_actors = [(name, pop) for name, pop in self.db.cursor.fetchall()]
                    self.db.store_statistic('tmdb', 'top_actors', json.dumps(top_actors))
            except Exception as e:
                logger.error(f"Error getting top actors: {e}")

            # Get most common genres if available
            try:
                self.db.cursor.execute(
                    "SELECT genre, COUNT(*) as count FROM tmdb_genres GROUP BY genre ORDER BY count DESC LIMIT 10"
                )
                top_genres = [(genre, count) for genre, count in self.db.cursor.fetchall()]
                self.db.store_statistic('tmdb', 'top_genres', json.dumps(top_genres))
            except Exception as e:
                logger.error(f"Error getting top genres: {e}")

            logger.info("TMDB statistics calculation completed")

        except Exception as e:
            logger.error(f"Error in TMDB statistics calculation: {e}")
            # Store minimal statistics to avoid errors later
            self.db.store_statistic('tmdb', 'data_status', "Statistics calculation failed")

    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics from all datasets"""
        stats = {}

        # Collect statistics from database
        for category in ['movielens', 'netflix', 'tmdb']:
            stats[category] = self.db.get_statistics(category)

        # Add ratings distribution
        stats['ratings_distribution'] = self.analyze_ratings_distribution()

        return stats

    def analyze_ratings_distribution(self) -> Dict:
        """Analyze ratings distribution across different datasets"""
        distributions = {}

        # MovieLens ratings distribution
        movielens_stats = self.db.execute_query(
            "SELECT "
            "AVG(rating) as mean, "
            "MIN(rating) as min, "
            "MAX(rating) as max, "
            "COUNT(*) as count "
            "FROM (SELECT rating FROM movielens_ratings ORDER BY RANDOM() LIMIT 1000000)"
        )

        if movielens_stats:
            mean, min_val, max_val, count = movielens_stats[0]
            total_count = self.db.execute_query("SELECT COUNT(*) FROM movielens_ratings")[0][0]

            distributions['movielens'] = {
                'mean': mean,
                'min': min_val,
                'max': max_val,
                'sample_count': count,
                'total_count': total_count
            }

        # Netflix ratings distribution
        netflix_stats = self.db.execute_query(
            "SELECT "
            "AVG(rating) as mean, "
            "MIN(rating) as min, "
            "MAX(rating) as max, "
            "COUNT(*) as count "
            "FROM (SELECT rating FROM netflix_ratings ORDER BY RANDOM() LIMIT 1000000)"
        )

        if netflix_stats:
            mean, min_val, max_val, count = netflix_stats[0]
            total_count = self.db.execute_query("SELECT COUNT(*) FROM netflix_ratings")[0][0]

            distributions['netflix'] = {
                'mean': mean,
                'min': min_val,
                'max': max_val,
                'sample_count': count,
                'total_count': total_count
            }

        return distributions

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

    def create_datasets(
            self,
            tokenizer: PreTrainedTokenizer,
            max_length: int = 512,
            train_size: float = 0.8,
            val_size: float = 0.1,
            test_size: float = 0.1,
            random_state: int = 42,
            max_examples: int = None
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create training, validation, and test datasets from the processed movie data in the database.

        Args:
            tokenizer: The tokenizer to use for encoding the texts
            max_length: Maximum sequence length for the model
            train_size: Proportion of data to use for training
            val_size: Proportion of data to use for validation
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            max_examples: Maximum number of examples to use (None for all available)

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Preparing training texts from database...")

        # Get training texts from database
        all_texts = self.db.create_training_texts(limit=max_examples)

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


# Modified Training Config
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
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 512):
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

        # Move model to appropriate device
        self.model = self.model.to(self.device)

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

    def _maybe_adjust_batch_size(self, oom_detected=False):
        """Dynamically adjust batch size based on available memory or OOM errors"""
        if not self.config.dynamic_batch_size:
            return

        if oom_detected:
            # Reduce batch size on OOM
            new_batch_size = max(1, self.current_batch_size // 2)
            logger.warning(f"OOM detected. Reducing batch size from {self.current_batch_size} to {new_batch_size}")
            self.current_batch_size = new_batch_size
            return True

        # Check if we should try to increase batch size
        if torch.cuda.is_available() and self.global_step > 0 and self.global_step % 50 == 0:
            # Get current utilization
            device_idx = self.device.index if self.device.index is not None else 0
            allocated = torch.cuda.memory_allocated(device_idx)
            total = torch.cuda.get_device_properties(device_idx).total_memory
            utilization = allocated / total

            # If utilization is low and we're below original batch size, try increasing
            if utilization < 0.7 and self.current_batch_size < self.original_batch_size:
                new_batch_size = min(self.current_batch_size + 2, self.original_batch_size)
                logger.info(
                    f"Memory utilization is {utilization:.2f}. Increasing batch size from {self.current_batch_size} to {new_batch_size}")
                self.current_batch_size = new_batch_size
                return True

        return False

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
                # Log initial system info
                if torch.cuda.is_available():
                    device_idx = self.device.index if self.device.index is not None else 0
                    device_name = torch.cuda.get_device_name(device_idx)
                    total_memory = torch.cuda.get_device_properties(device_idx).total_memory / 1e9

                    wandb.log({
                        "gpu_name": device_name,
                        "gpu_memory_gb": total_memory,
                        "cuda_version": torch.version.cuda,
                    })
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

    def save_checkpoint(self, val_loss: float, is_best: bool = False, is_emergency: bool = False) -> None:
        """Save model checkpoint with proper error handling for CUDA devices"""
        try:
            # First move model to CPU for saving (prevents CUDA OOM errors during saving)
            model_state_dict = {k: v.to("cpu") for k, v in self.model.state_dict().items()}

            checkpoint = {
                'epoch': self.global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'val_loss': val_loss,
                'config': vars(self.config),
            }

            # Save EMA state if used
            if self.ema is not None:
                checkpoint['ema_state'] = {k: v.to("cpu") for k, v in self.ema.shadow.items()}

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
            del checkpoint, model_state_dict
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
                # Move data to device
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
                        # Move batch to device
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