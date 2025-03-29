import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, PreTrainedModel
import random
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecommendationConfig:
    """Configuration for movie recommendation system"""
    content_weight: float = 0.4  # Weight for content-based recommendations
    collab_weight: float = 0.4  # Weight for collaborative filtering recommendations
    llm_weight: float = 0.2  # Weight for LLM-based recommendations
    top_k: int = 10  # Number of recommendations to return
    diversity_factor: float = 0.2  # Factor to increase diversity in recommendations
    min_rating_threshold: float = 3.5  # Minimum rating threshold for recommendations
    use_cuda: bool = True  # Whether to use CUDA for LLM inference


class MovieRecommender:
    """Movie recommendation system that combines collaborative filtering,
    content-based filtering, and LLM-based recommendations"""

    def __init__(
            self,
            processor,  # MovieDataProcessor instance with loaded data
            model: PreTrainedModel,  # Trained language model
            tokenizer: PreTrainedTokenizer,
            config: RecommendationConfig = None
    ):
        self.processor = processor
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RecommendationConfig()

        # Prepare device for model inference
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_cuda else "cpu")
        self.model = self.model.to(self.device)

        # Initialize recommendation components
        self._initialize_recommendation_components()

    def _initialize_recommendation_components(self):
        """Initialize various recommendation components based on available data"""
        logger.info("Initializing recommendation components...")

        # Check if we have MovieLens data for content-based filtering
        if self.processor.movielens_data and 'movies' in self.processor.movielens_data:
            self._prepare_content_features()
            logger.info("Content-based features prepared successfully")
        else:
            logger.warning("MovieLens data not available, content-based filtering will be limited")

        # Check if we have ratings data for collaborative filtering
        if self.processor.movielens_data and 'ratings' in self.processor.movielens_data:
            self._prepare_user_item_matrix()
            logger.info("User-item matrix prepared successfully for collaborative filtering")
        else:
            logger.warning("Ratings data not available, collaborative filtering will be limited")

        # Ensure model is ready for inference
        self.model.eval()
        logger.info("LLM prepared for generative recommendations")

    def _prepare_content_features(self):
        """Prepare content-based features from movies metadata"""
        try:
            # Get movies data from MovieLens
            movies_df = self.processor.movielens_data['movies']

            # Create genre features as one-hot encoding
            all_genres = set()
            for genres_list in movies_df['genres']:
                if isinstance(genres_list, list):
                    all_genres.update(genres_list)

            # Create genre feature matrix
            self.genre_features = np.zeros((len(movies_df), len(all_genres)))
            self.genre_labels = list(all_genres)

            for i, genres_list in enumerate(movies_df['genres']):
                if isinstance(genres_list, list):
                    for genre in genres_list:
                        if genre in self.genre_labels:
                            genre_idx = self.genre_labels.index(genre)
                            self.genre_features[i, genre_idx] = 1

            # Store movie IDs for reference
            self.movieId_to_idx = {movieId: i for i, movieId in enumerate(movies_df['movieId'])}
            self.idx_to_movieId = {i: movieId for movieId, i in self.movieId_to_idx.items()}

            # Calculate similarity matrix for content-based filtering
            self.content_similarity = cosine_similarity(self.genre_features)

            logger.info(f"Content features prepared for {len(movies_df)} movies")

        except Exception as e:
            logger.error(f"Error preparing content features: {e}")
            # Create empty features as fallback
            self.genre_features = np.array([])
            self.genre_labels = []
            self.content_similarity = np.array([])
            self.movieId_to_idx = {}
            self.idx_to_movieId = {}

    def _prepare_user_item_matrix(self):
        """Prepare user-item matrix for collaborative filtering"""
        try:
            # Get ratings data
            ratings_df = self.processor.movielens_data['ratings']

            # Create pivot table for user-item matrix
            user_item_matrix = ratings_df.pivot_table(
                index='userId',
                columns='movieId',
                values='rating',
                fill_value=0
            )

            # Store matrix for collaborative filtering
            self.user_item_matrix = user_item_matrix.values
            self.user_id_mapping = {userId: i for i, userId in enumerate(user_item_matrix.index)}
            self.movie_id_mapping = {movieId: i for i, movieId in enumerate(user_item_matrix.columns)}
            self.inv_movie_id_mapping = {i: movieId for movieId, i in self.movie_id_mapping.items()}

            # Calculate similarity matrix for item-based collaborative filtering
            self.item_similarity = cosine_similarity(self.user_item_matrix.T)

            logger.info(f"User-item matrix prepared with shape {self.user_item_matrix.shape}")

        except Exception as e:
            logger.error(f"Error preparing user-item matrix: {e}")
            # Create empty matrix as fallback
            self.user_item_matrix = np.array([])
            self.user_id_mapping = {}
            self.movie_id_mapping = {}
            self.inv_movie_id_mapping = {}
            self.item_similarity = np.array([])

    def generate_text_for_movie(self, movie_info: Dict) -> str:
        """Generate descriptive text for a movie using the trained LLM"""
        try:
            # Prepare prompt for the model
            if 'title' in movie_info and 'genres' in movie_info:
                genres_str = ' | '.join(movie_info['genres']) if isinstance(movie_info['genres'], list) else movie_info[
                    'genres']
                prompt = f"Movie Title: {movie_info['title']}\nGenres: {genres_str}\n\nDescription:"
            else:
                prompt = f"Describe a movie like {movie_info.get('title', 'this')}"

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate text
            with torch.no_grad():
                output = self.model.generate(
                    inputs["input_ids"],
                    max_length=150,
                    temperature=0.8,
                    top_p=0.9,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and return generated text
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract just the description part (after the prompt)
            if "Description:" in generated_text:
                description = generated_text.split("Description:")[1].strip()
                return description

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text for movie: {e}")
            return ""

    def get_content_based_recommendations(self, movie_ids: List[int], n: int = 10) -> List[Dict]:
        """Get content-based recommendations based on movie IDs"""
        if not self.content_similarity.size or not movie_ids:
            logger.warning("Content-based filtering unavailable or no input movies provided")
            return []

        try:
            # Find movie indices
            movie_indices = [self.movieId_to_idx.get(movie_id, -1) for movie_id in movie_ids]
            movie_indices = [idx for idx in movie_indices if idx >= 0]

            if not movie_indices:
                logger.warning("None of the provided movie IDs were found in the content database")
                return []

            # Calculate average similarity to all input movies
            sim_scores = np.zeros(self.content_similarity.shape[0])
            for idx in movie_indices:
                sim_scores += self.content_similarity[idx]

            sim_scores = sim_scores / len(movie_indices)

            # Get top similar movies
            movie_scores = [(i, score) for i, score in enumerate(sim_scores) if i not in movie_indices]
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            movie_indices = [i for i, _ in movie_scores[:n]]

            # Get movie details
            movies_df = self.processor.movielens_data['movies']
            recommendations = []

            for idx in movie_indices:
                movie_id = self.idx_to_movieId[idx]
                movie_info = movies_df.loc[movies_df['movieId'] == movie_id].iloc[0].to_dict()

                recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'score': float(sim_scores[idx]),
                    'source': 'content'
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []

    def get_collaborative_recommendations(self, movie_ids: List[int], n: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations based on movie IDs"""
        if not self.item_similarity.size or not movie_ids:
            logger.warning("Collaborative filtering unavailable or no input movies provided")
            return []

        try:
            # Find movie indices in the item-similarity matrix
            movie_indices = [self.movie_id_mapping.get(movie_id, -1) for movie_id in movie_ids]
            movie_indices = [idx for idx in movie_indices if idx >= 0]

            if not movie_indices:
                logger.warning("None of the provided movie IDs were found in the collaborative filtering database")
                return []

            # Calculate average similarity to all input movies
            sim_scores = np.zeros(self.item_similarity.shape[0])
            for idx in movie_indices:
                sim_scores += self.item_similarity[idx]

            sim_scores = sim_scores / len(movie_indices)

            # Get top similar movies
            movie_scores = [(i, score) for i, score in enumerate(sim_scores) if i not in movie_indices]
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [i for i, _ in movie_scores[:n]]

            # Get movie details
            movies_df = self.processor.movielens_data['movies']
            recommendations = []

            for idx in top_indices:
                movie_id = self.inv_movie_id_mapping[idx]
                if movie_id in movies_df['movieId'].values:
                    movie_info = movies_df.loc[movies_df['movieId'] == movie_id].iloc[0].to_dict()

                    recommendations.append({
                        'movieId': int(movie_id),
                        'title': movie_info['title'],
                        'genres': movie_info.get('genres', []),
                        'score': float(sim_scores[idx]),
                        'source': 'collaborative'
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []

    def get_llm_recommendations(self, movie_ids: List[int], n: int = 5) -> List[Dict]:
        """Generate recommendations by prompting the language model"""
        if not movie_ids:
            logger.warning("No input movies provided for LLM recommendations")
            return []

        try:
            # Get movie details for the inputs
            movies_df = self.processor.movielens_data['movies']
            input_movies = []

            for movie_id in movie_ids:
                if movie_id in movies_df['movieId'].values:
                    movie = movies_df.loc[movies_df['movieId'] == movie_id].iloc[0]
                    input_movies.append({
                        'title': movie['title'],
                        'genres': movie['genres']
                    })

            if not input_movies:
                logger.warning("None of the provided movie IDs were found in the database")
                return []

            # Prepare prompt for the model with movie titles and genres
            movie_descriptions = []
            for movie in input_movies:
                genres_str = ' | '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
                movie_descriptions.append(f"- {movie['title']} ({genres_str})")

            prompt = (
                    "Based on these movies:\n" +
                    "\n".join(movie_descriptions) +
                    "\n\nRecommend five similar movies with titles and genres in this format:\n" +
                    "1. Movie Title (Genre1 | Genre2)\n"
            )

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate text
            with torch.no_grad():
                output = self.model.generate(
                    inputs["input_ids"],
                    max_length=300,
                    temperature=0.8,
                    top_p=0.9,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode generated text
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Parse recommended movies from generated text
            recommendations = []

            # Extract just the recommendations part (after the prompt)
            if "similar movies" in generated_text:
                rec_text = generated_text.split("similar movies")[1].strip()

                # Process each line
                import re
                movie_matches = re.findall(r'(?:\d+\.\s+)?(.*?)\s+\((.*?)\)', rec_text)

                for i, (title, genres_str) in enumerate(movie_matches):
                    if i >= n:
                        break

                    genres = [g.strip() for g in genres_str.split('|')]

                    recommendations.append({
                        'movieId': -1,  # No real movie ID for LLM-generated recommendations
                        'title': title.strip(),
                        'genres': genres,
                        'score': 1.0 - (i * 0.1),  # Decreasing score based on position
                        'source': 'llm',
                        'description': self.generate_text_for_movie({'title': title, 'genres': genres})
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Error in LLM recommendations: {e}")
            return []

    def enhance_recommendations_with_descriptions(self, recommendations: List[Dict]) -> List[Dict]:
        """Add LLM-generated descriptions to recommendations"""
        enhanced_recommendations = []

        for rec in tqdm(recommendations, desc="Generating descriptions"):
            if 'description' not in rec:
                rec['description'] = self.generate_text_for_movie(rec)
            enhanced_recommendations.append(rec)

        return enhanced_recommendations

    def diversify_recommendations(self, recommendations: List[Dict], diversity_factor: float = None) -> List[Dict]:
        """Apply diversity factor to ensure varied recommendations"""
        if not recommendations or len(recommendations) <= 1:
            return recommendations

        diversity_factor = diversity_factor or self.config.diversity_factor

        # Group recommendations by genre
        genre_groups = {}
        for rec in recommendations:
            genres = rec.get('genres', [])
            if isinstance(genres, str):
                genres = [genres]

            for genre in genres:
                if genre not in genre_groups:
                    genre_groups[genre] = []
                genre_groups[genre].append(rec)

        # Rerank recommendations to promote diversity
        seen_recs = set()
        diversified_recs = []

        # First, ensure we have at least one movie from each major genre
        major_genres = sorted(genre_groups.keys(), key=lambda g: len(genre_groups[g]), reverse=True)

        for genre in major_genres:
            if len(diversified_recs) >= len(recommendations):
                break

            best_rec = None
            best_score = -1

            for rec in genre_groups[genre]:
                rec_id = rec.get('movieId', -1)
                if rec_id not in seen_recs and rec.get('score', 0) > best_score:
                    best_rec = rec
                    best_score = rec.get('score', 0)

            if best_rec:
                diversified_recs.append(best_rec)
                seen_recs.add(best_rec.get('movieId', -1))

        # Then fill in with the highest scoring remaining recommendations
        remaining_recs = [rec for rec in recommendations if rec.get('movieId', -1) not in seen_recs]
        remaining_recs.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Fill up to original length
        while len(diversified_recs) < len(recommendations) and remaining_recs:
            next_rec = remaining_recs.pop(0)
            diversified_recs.append(next_rec)
            seen_recs.add(next_rec.get('movieId', -1))

        return diversified_recs

    def recommend(self, movie_ids: List[int], n: int = None) -> List[Dict]:
        """Main recommendation function that combines all methods"""
        n = n or self.config.top_k

        logger.info(f"Generating recommendations based on {len(movie_ids)} movies")

        # Get recommendations from each method
        content_recs = self.get_content_based_recommendations(movie_ids, n=n * 2)
        collab_recs = self.get_collaborative_recommendations(movie_ids, n=n * 2)
        llm_recs = self.get_llm_recommendations(movie_ids, n=max(3, int(n * 0.5)))

        logger.info(
            f"Generated {len(content_recs)} content-based, {len(collab_recs)} collaborative, and {len(llm_recs)} LLM recommendations")

        # Score normalization for each method
        for recs in [content_recs, collab_recs, llm_recs]:
            if recs:
                max_score = max(rec.get('score', 0) for rec in recs)
                min_score = min(rec.get('score', 0) for rec in recs)
                score_range = max_score - min_score if max_score > min_score else 1.0

                for rec in recs:
                    normalized_score = (rec.get('score', 0) - min_score) / score_range
                    rec['normalized_score'] = normalized_score

        # Combine all recommendations
        all_recs = []
        all_recs.extend(content_recs)
        all_recs.extend(collab_recs)
        all_recs.extend(llm_recs)

        # Apply weights to each method
        for rec in all_recs:
            if rec['source'] == 'content':
                rec['weighted_score'] = rec.get('normalized_score', 0) * self.config.content_weight
            elif rec['source'] == 'collaborative':
                rec['weighted_score'] = rec.get('normalized_score', 0) * self.config.collab_weight
            elif rec['source'] == 'llm':
                rec['weighted_score'] = rec.get('normalized_score', 0) * self.config.llm_weight

        # Remove duplicates, preferring higher-scored recommendations
        unique_recs = {}
        for rec in all_recs:
            title = rec.get('title', '').lower()
            if title not in unique_recs or rec.get('weighted_score', 0) > unique_recs[title].get('weighted_score', 0):
                unique_recs[title] = rec

        # Convert back to list and sort by weighted score
        recommendations = list(unique_recs.values())
        recommendations.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)

        # Apply diversity factor
        recommendations = self.diversify_recommendations(recommendations)

        # Truncate to requested number
        recommendations = recommendations[:n]

        # Add descriptions for the final recommendations
        recommendations = self.enhance_recommendations_with_descriptions(recommendations)

        logger.info(f"Returning {len(recommendations)} final recommendations")
        return recommendations


# Example usage
def example_recommendation_flow(model_path="./movie_llm_checkpoints/best_model.pt"):
    """Example of how to use the recommendation system with the trained model"""
    try:
        # Initialize data processor
        processor = MovieDataProcessor()
        processor.load_all_data()  # Load all datasets

        # Load trained model
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        config = checkpoint['config']

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint['model_state_dict'])

        # Initialize recommender
        rec_config = RecommendationConfig(
            content_weight=0.4,
            collab_weight=0.4,
            llm_weight=0.2,
            top_k=10
        )

        recommender = MovieRecommender(
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            config=rec_config
        )

        # Get recommendations
        example_movies = [122, 1893, 79132]  # Example movie IDs from MovieLens
        recommendations = recommender.recommend(example_movies)

        # Display recommendations
        print(f"\nRecommendations based on movie IDs: {example_movies}")
        for i, rec in enumerate(recommendations):
            print(f"{i + 1}. {rec['title']} ({', '.join(rec['genres'])})")
            print(f"   Source: {rec['source']}, Score: {rec.get('weighted_score', 0):.3f}")
            if 'description' in rec:
                print(f"   Description: {rec['description'][:100]}...")
            print()

        return recommendations

    except Exception as e:
        logger.error(f"Error in example recommendation flow: {e}")
        raise