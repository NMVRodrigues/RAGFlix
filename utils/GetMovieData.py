import requests
import pandas as pd
import gzip
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TMDBDownloader:
    """Downloads data from TMDB API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        url = f"{self.base_url}/{endpoint}"
        request_params = {'api_key': self.api_key}
        
        if params:
            request_params.update(params)
        
        try:
            response = self.session.get(url, params=request_params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def discover_by_date_range(self, 
                              media_type: str,
                              start_date: str, 
                              end_date: str, 
                              max_pages: int = 500) -> List[Dict]:
        """
        Discover movies or TV shows by release date range
        
        Args:
            media_type: 'movie' or 'tv'
            start_date: 'YYYY-MM-DD' format
            end_date: 'YYYY-MM-DD' format
            max_pages: Maximum pages to fetch
        """
        logger.info(f"Discovering {media_type}s from {start_date} to {end_date}")
        
        all_items = []
        page = 1
        
        date_param = 'primary_release_date.gte' if media_type == 'movie' else 'first_air_date.gte'
        date_param_end = 'primary_release_date.lte' if media_type == 'movie' else 'first_air_date.lte'
        
        while page <= max_pages:
            params = {
                'page': page,
                date_param: start_date,
                date_param_end: end_date,
                'sort_by': 'popularity.desc'
            }
            
            data = self._request(f'discover/{media_type}', params)
            
            if not data or 'results' not in data:
                break
                
            results = data['results']
            if not results:
                break
                
            all_items.extend(results)
            
            if page >= data.get('total_pages', 0):
                break
                
            page += 1
            time.sleep(0.25)  # Rate limiting
        
        logger.info(f"Found {len(all_items)} {media_type}s")
        return all_items
    
    def get_detailed_info(self, media_type: str, item_ids: List[int]) -> List[Dict]:
        """Get detailed information for movies or TV shows"""
        logger.info(f"Fetching details for {len(item_ids)} {media_type}s")
        
        detailed_items = []
        
        for item_id in item_ids:
            endpoint = f"{media_type}/{item_id}"
            params = {'append_to_response': 'credits,keywords,videos,external_ids'}
            
            details = self._request(endpoint, params)
            if details:
                detailed_items.append(details)
            
            time.sleep(0.25)  # Rate limiting
            
            if len(detailed_items) % 100 == 0:
                logger.info(f"Processed {len(detailed_items)} items")
        
        logger.info(f"Retrieved details for {len(detailed_items)} {media_type}s")
        return detailed_items


class IMDbDownloader:
    """Downloads and processes IMDb datasets"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.base_url = "https://datasets.imdbws.com"
        self.datasets = {
            'title_basics': 'title.basics.tsv.gz',
            'title_ratings': 'title.ratings.tsv.gz',
            'title_crew': 'title.crew.tsv.gz',
            'title_principals': 'title.principals.tsv.gz',
            'title_episode': 'title.episode.tsv.gz',
            'name_basics': 'name.basics.tsv.gz'
        }
        
        self.data_dir.mkdir(exist_ok=True)
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download specific IMDb dataset"""
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        filename = self.datasets[dataset_name]
        url = f"{self.base_url}/{filename}"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"Dataset already exists: {filename}")
            return True
        
        logger.info(f"Downloading {filename}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")
            return False
    
    def load_dataset(self, dataset_name: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load IMDb dataset into DataFrame"""
        filename = self.datasets[dataset_name]
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            if not self.download_dataset(dataset_name):
                raise FileNotFoundError(f"Could not download {filename}")
        
        logger.info(f"Loading {filename}")
        
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f, sep='\t', na_values=['\\N'], nrows=nrows, low_memory=False)
        
        logger.info(f"Loaded {len(df)} rows from {filename}")
        return df
    
    def get_titles_by_date_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Get movies and TV shows within year range"""
        logger.info(f"Loading titles from {start_year} to {end_year}")
        
        basics = self.load_dataset('title_basics')
        
        # Convert startYear to numeric
        basics['startYear'] = pd.to_numeric(basics['startYear'], errors='coerce')
        
        # Filter by date range and title types
        title_types = ['movie', 'tvSeries', 'tvMiniSeries']
        filtered = basics[
            (basics['titleType'].isin(title_types)) &
            (basics['startYear'] >= start_year) &
            (basics['startYear'] <= end_year)
        ].copy()
        
        logger.info(f"Found {len(filtered)} titles in date range")
        return filtered


class DataProcessor:
    """Processes and combines datasets"""
    
    @staticmethod
    def process_tmdb_movies(movies: List[Dict]) -> pd.DataFrame:
        """Process TMDB movie data into structured format"""
        processed = []
        
        for movie in movies:
            credits = movie.get('credits', {})
            cast = credits.get('cast', [])
            crew = credits.get('crew', [])
            
            # Extract key crew roles
            director = next((p['name'] for p in crew if p.get('job') == 'Director'), None)
            writers = [p['name'] for p in crew if p.get('job') in ['Writer', 'Screenplay']]
            producers = [p['name'] for p in crew if p.get('job') == 'Producer']
            
            processed_movie = {
                'tmdb_id': movie.get('id'),
                'imdb_id': movie.get('imdb_id'),
                'title': movie.get('title'),
                'overview': movie.get('overview'),
                'release_date': movie.get('release_date'),
                'runtime': movie.get('runtime'),
                'budget': movie.get('budget', 0),
                'revenue': movie.get('revenue', 0),
                'vote_average': movie.get('vote_average'),
                'vote_count': movie.get('vote_count'),
                'popularity': movie.get('popularity'),
                'genres': [g['name'] for g in movie.get('genres', [])],
                'keywords': [k['name'] for k in movie.get('keywords', {}).get('keywords', [])],
                'cast': [actor['name'] for actor in cast[:10]],
                'director': director,
                'writers': writers,
                'producers': producers[:3],
                'poster_path': movie.get('poster_path'),
                'backdrop_path': movie.get('backdrop_path'),
                'original_language': movie.get('original_language'),
                'production_countries': [c['name'] for c in movie.get('production_countries', [])]
            }
            processed.append(processed_movie)
        
        return pd.DataFrame(processed)
    
    @staticmethod
    def process_tmdb_tv_shows(shows: List[Dict]) -> pd.DataFrame:
        """Process TMDB TV show data into structured format"""
        processed = []
        
        for show in shows:
            credits = show.get('credits', {})
            cast = credits.get('cast', [])
            crew = credits.get('crew', [])
            
            # Extract creators and key crew
            creators = [c['name'] for c in show.get('created_by', [])]
            producers = [p['name'] for p in crew if p.get('job') == 'Executive Producer']
            
            processed_show = {
                'tmdb_id': show.get('id'),
                'imdb_id': show.get('external_ids', {}).get('imdb_id'),
                'name': show.get('name'),
                'overview': show.get('overview'),
                'first_air_date': show.get('first_air_date'),
                'last_air_date': show.get('last_air_date'),
                'number_of_seasons': show.get('number_of_seasons'),
                'number_of_episodes': show.get('number_of_episodes'),
                'episode_run_time': show.get('episode_run_time'),
                'vote_average': show.get('vote_average'),
                'vote_count': show.get('vote_count'),
                'popularity': show.get('popularity'),
                'genres': [g['name'] for g in show.get('genres', [])],
                'keywords': [k['name'] for k in show.get('keywords', {}).get('keywords', [])],
                'cast': [actor['name'] for actor in cast[:10]],
                'creators': creators,
                'producers': producers[:3],
                'networks': [n['name'] for n in show.get('networks', [])],
                'poster_path': show.get('poster_path'),
                'backdrop_path': show.get('backdrop_path'),
                'original_language': show.get('original_language'),
                'origin_country': show.get('origin_country', []),
                'status': show.get('status')
            }
            processed.append(processed_show)
        
        return pd.DataFrame(processed)


class MovieDatasetDownloader:
    """Main downloader class"""
    
    def __init__(self, tmdb_api_key: str, data_dir: str = "./movie_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.tmdb = TMDBDownloader(tmdb_api_key)
        self.imdb = IMDbDownloader(self.data_dir / "imdb")
        self.processor = DataProcessor()
    
    def download_by_date_range(self, 
                              start_date: str, 
                              end_date: str,
                              include_movies: bool = True,
                              include_tv: bool = True,
                              get_details: bool = True,
                              max_items: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Download movies and TV shows within date range
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            include_movies: Whether to download movies
            include_tv: Whether to download TV shows
            get_details: Whether to fetch detailed information
            max_items: Maximum items per category
        """
        results = {}
        
        if include_movies:
            logger.info("Downloading movies")
            movies = self.tmdb.discover_by_date_range('movie', start_date, end_date)
            
            if max_items:
                movies = movies[:max_items]
            
            if get_details and movies:
                movie_ids = [m['id'] for m in movies]
                detailed_movies = self.tmdb.get_detailed_info('movie', movie_ids)
                movies_df = self.processor.process_tmdb_movies(detailed_movies)
            else:
                movies_df = pd.DataFrame(movies)
            
            # Save movies
            output_file = self.data_dir / f"movies_{start_date}_{end_date}.csv"
            movies_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(movies_df)} movies to {output_file}")
            results['movies'] = movies_df
        
        if include_tv:
            logger.info("Downloading TV shows")
            tv_shows = self.tmdb.discover_by_date_range('tv', start_date, end_date)
            
            if max_items:
                tv_shows = tv_shows[:max_items]
            
            if get_details and tv_shows:
                tv_ids = [tv['id'] for tv in tv_shows]
                detailed_tv = self.tmdb.get_detailed_info('tv', tv_ids)
                tv_df = self.processor.process_tmdb_tv_shows(detailed_tv)
            else:
                tv_df = pd.DataFrame(tv_shows)
            
            # Save TV shows
            output_file = self.data_dir / f"tv_shows_{start_date}_{end_date}.csv"
            tv_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(tv_df)} TV shows to {output_file}")
            results['tv_shows'] = tv_df
        
        return results
    
    def download_imdb_datasets(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Download and filter IMDb datasets by year range"""
        logger.info("Downloading IMDb datasets")
        
        # Download core datasets
        for dataset in ['title_basics', 'title_ratings', 'title_crew']:
            self.imdb.download_dataset(dataset)
        
        # Get filtered titles
        titles = self.imdb.get_titles_by_date_range(start_year, end_year)
        
        # Load ratings and merge
        ratings = self.imdb.load_dataset('title_ratings')
        combined = pd.merge(titles, ratings, on='tconst', how='left')
        
        # Save combined data
        output_file = self.data_dir / f"imdb_titles_{start_year}_{end_year}.csv"
        combined.to_csv(output_file, index=False)
        logger.info(f"Saved {len(combined)} IMDb titles to {output_file}")
        
        return combined


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download movie and TV datasets")
    parser.add_argument("--api-key", required=True, help="TMDB API key")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="./movie_data", help="Data directory")
    parser.add_argument("--no-movies", action="store_true", help="Skip movies")
    parser.add_argument("--no-tv", action="store_true", help="Skip TV shows")
    parser.add_argument("--no-details", action="store_true", help="Skip detailed info")
    parser.add_argument("--max-items", type=int, help="Max items per category")
    parser.add_argument("--include-imdb", action="store_true", help="Include IMDb data")
    
    args = parser.parse_args()

    downloader = MovieDatasetDownloader(args.api_key, args.data_dir)

    # Download TMDB data
    results = downloader.download_by_date_range(
        start_date=args.start_date,
        end_date=args.end_date,
        include_movies=not args.no_movies,
        include_tv=not args.no_tv,
        get_details=not args.no_details,
        max_items=args.max_items
    )
    
    # Download IMDb data if requested
    if args.include_imdb:
        start_year = int(args.start_date.split('-')[0])
        end_year = int(args.end_date.split('-')[0])
        imdb_data = downloader.download_imdb_datasets(start_year, end_year)
        results['imdb'] = imdb_data
    
    logger.info("Download complete")
    for key, df in results.items():
        logger.info(f"{key}: {len(df)} records")


if __name__ == "__main__":
    main()