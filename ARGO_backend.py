import os
import pickle
import pandas as pd
import numpy as np
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import sqlite3
from dataclasses import dataclass
import warnings
import requests
import time
from urllib.parse import quote

# Suppress warnings
warnings.filterwarnings("ignore")

# Try to import llama-cpp-python for Phi-3.5 model
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ARGORecord:
    """Data class for ARGO measurements"""
    float_id: str
    date: str
    latitude: float
    longitude: float
    depth: float
    temperature: float
    salinity: float
    pressure: float

class DuckDuckGoSearch:
    """Simple DuckDuckGo search integration"""
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
        self.instant_answer_url = "https://api.duckduckgo.com/"
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search DuckDuckGo and return results"""
        try:
            # Use DuckDuckGo Instant Answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(self.instant_answer_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Extract abstract
            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractText', 'DuckDuckGo Result'),
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', ''),
                    'source': 'DuckDuckGo Instant Answer'
                })
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('FirstURL', {}).get('Text', 'Related Topic'),
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', {}).get('FirstURL', ''),
                        'source': 'DuckDuckGo Related'
                    })
            
            # If no results, try a different approach
            if not results:
                # Fallback: create a generic response
                results.append({
                    'title': f'Search results for: {query}',
                    'snippet': f'I searched for "{query}" but couldn\'t find specific instant answers. This may be a complex topic requiring detailed analysis.',
                    'url': f'https://duckduckgo.com/?q={quote(query)}',
                    'source': 'DuckDuckGo Search'
                })
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return [{
                'title': 'Search Error',
                'snippet': f'Could not perform web search: {str(e)}',
                'url': '',
                'source': 'Error'
            }]

class SimpleTextSearch:
    """Simple text-based search without heavy ML dependencies"""
    
    def __init__(self, documents: List[str], metadata: List[Dict]):
        self.documents = documents
        self.metadata = metadata
        self.index = self._create_index()
    
    def _create_index(self) -> Dict[str, List[int]]:
        """Create simple keyword index"""
        index = {}
        for i, doc in enumerate(self.documents):
            words = re.findall(r'\b\w+\b', doc.lower())
            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(i)
        return index
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, Dict]]:
        """Simple keyword-based search"""
        query_words = re.findall(r'\b\w+\b', query.lower())
        scores = {}
        
        for word in query_words:
            if word in self.index:
                for doc_idx in self.index[word]:
                    scores[doc_idx] = scores.get(doc_idx, 0) + 1
        
        # Sort by score and return top k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_idx, score in sorted_docs:
            results.append((self.documents[doc_idx], self.metadata[doc_idx]))
        
        return results

class Phi35Model:
    """Phi-3.5-mini-instruct model wrapper"""
    
    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the Phi-3.5 model"""
        try:
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python not available")
            
            model_file = Path(self.model_path) / "Phi-3.5-mini-instruct-Q4_K_M.gguf"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            self.model = Llama(
                model_path=str(model_file),
                n_ctx=4096,  # Context window
                n_threads=4,  # Number of threads
                verbose=False
            )
            logger.info("Phi-3.5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Phi-3.5 model: {e}")
            self.model = None
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using Phi-3.5 model"""
        if self.model is None:
            return "AI model not available. Using fallback response generation."
        
        try:
            # Format prompt for Phi-3.5-mini-instruct
            formatted_prompt = f"""<|user|>
{prompt}<|end|>
<|assistant|>
"""
            
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["<|end|>", "<|user|>"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            return f"Error generating AI response: {str(e)}"

class EnhancedARGOChatbot:
    """Enhanced ARGO Chatbot with AI and web search capabilities"""
    
    def __init__(self, data_dir: str = ".", models_dir: str = "./models"):
        """Initialize Enhanced ARGO Chatbot"""
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.df = None
        self.db_path = self.data_dir / "argo_data.db"
        self.search_engine = None
        self.web_search = DuckDuckGoSearch()
        self.ai_model = Phi35Model(self.models_dir)
        
        logger.info("Starting Enhanced ARGO Chatbot initialization...")
        self._load_data()
        self._setup_database()
        self._setup_search()
        logger.info("Enhanced ARGO Chatbot initialized successfully")
    
    def _load_data(self):
        """Load ARGO dataset with error handling"""
        try:
            csv_path = self.data_dir / "argo_dummy.csv"
            if not csv_path.exists():
                logger.info("Creating dummy ARGO data...")
                self._create_dummy_data(csv_path)
            
            self.df = pd.read_csv(csv_path)
            # Ensure date column is string for consistency
            self.df['date'] = pd.to_datetime(self.df['date']).dt.strftime('%Y-%m-%d')
            logger.info(f"Loaded {len(self.df)} records from ARGO dataset")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._create_dummy_data()
    
    def _create_dummy_data(self, csv_path: Optional[Path] = None):
        """Create realistic dummy ARGO data for testing"""
        if csv_path is None:
            csv_path = self.data_dir / "argo_dummy.csv"
        
        np.random.seed(42)
        n_records = 2000  # Increased dataset size
        
        # Generate realistic ARGO float data
        float_ids = [f"ARGO_{i:04d}" for i in range(1000, 1000 + n_records)]
        
        # Generate dates over 3 years with seasonal patterns
        start_date = pd.Timestamp('2021-01-01')
        dates = pd.date_range(start=start_date, periods=n_records, freq='D')
        
        # Generate realistic oceanographic data with patterns
        latitudes = np.random.uniform(-70, 70, n_records)
        longitudes = np.random.uniform(-180, 180, n_records)
        depths = np.random.exponential(200, n_records).clip(5, 2000)
        
        # Temperature varies with depth, latitude, and season
        day_of_year = dates.dayofyear
        seasonal_temp = 5 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal variation
        surface_temps = 20 - 0.5 * np.abs(latitudes) + seasonal_temp + np.random.normal(0, 2, n_records)
        depth_effect = -depths * 0.015  # Temperature decreases with depth
        temperatures = (surface_temps + depth_effect + np.random.normal(0, 1, n_records)).clip(-2, 35)
        
        # Salinity with realistic oceanic patterns
        base_salinity = 35 + 0.1 * np.abs(latitudes) * np.random.normal(1, 0.1, n_records)
        salinities = (base_salinity + np.random.normal(0, 1.5, n_records)).clip(30, 40)
        
        # Pressure correlates with depth
        pressures = depths * 0.1 + np.random.normal(0, 3, n_records)
        
        dummy_data = {
            'float_id': float_ids,
            'date': dates.strftime('%Y-%m-%d'),
            'latitude': latitudes.round(4),
            'longitude': longitudes.round(4),
            'depth': depths.round(1),
            'temperature': temperatures.round(2),
            'salinity': salinities.round(2),
            'pressure': pressures.round(1)
        }
        
        self.df = pd.DataFrame(dummy_data)
        self.df.to_csv(csv_path, index=False)
        logger.info(f"Created realistic dummy ARGO data with {len(self.df)} records")
    
    def _setup_database(self):
        """Setup SQLite database for fast queries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table with indices for better performance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS argo_data (
                    id INTEGER PRIMARY KEY,
                    float_id TEXT,
                    date TEXT,
                    latitude REAL,
                    longitude REAL,
                    depth REAL,
                    temperature REAL,
                    salinity REAL,
                    pressure REAL
                )
            ''')
            
            # Create indices for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_float_id ON argo_data(float_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON argo_data(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_location ON argo_data(latitude, longitude)')
            
            # Check if data exists
            cursor.execute('SELECT COUNT(*) FROM argo_data')
            count = cursor.fetchone()[0]
            
            if count == 0:
                # Insert data
                for _, row in self.df.iterrows():
                    cursor.execute('''
                        INSERT INTO argo_data 
                        (float_id, date, latitude, longitude, depth, temperature, salinity, pressure)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['float_id'], row['date'], row['latitude'], row['longitude'],
                        row['depth'], row['temperature'], row['salinity'], row['pressure']
                    ))
                
                conn.commit()
                logger.info(f"Inserted {len(self.df)} records into database")
            else:
                logger.info(f"Database already contains {count} records")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def _setup_search(self):
        """Setup simple text search with enhanced content"""
        try:
            documents = []
            metadata = []
            
            for _, row in self.df.iterrows():
                content = f"""
ARGO Float {row['float_id']} oceanographic measurement recorded on {row['date']}
Geographic location: latitude {row['latitude']} degrees, longitude {row['longitude']} degrees
Ocean profile data: measurement depth {row['depth']} meters, water temperature {row['temperature']} degrees Celsius, 
ocean salinity {row['salinity']} PSU (Practical Salinity Units), water pressure {row['pressure']} decibars
Oceanographic conditions from autonomous float {row['float_id']} collected {row['date']} at coordinates {row['latitude']}, {row['longitude']}
Marine data temperature salinity depth pressure ocean conditions float measurement
                """.strip()
                
                documents.append(content)
                metadata.append({
                    'float_id': row['float_id'],
                    'date': row['date'],
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'depth': float(row['depth']),
                    'temperature': float(row['temperature']),
                    'salinity': float(row['salinity']),
                    'pressure': float(row['pressure'])
                })
            
            self.search_engine = SimpleTextSearch(documents, metadata)
            logger.info(f"Initialized enhanced search with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error setting up search: {e}")
            raise
    
    def analyze_ocean_patterns(self, query_type: str) -> Dict[str, Any]:
        """Analyze patterns in oceanographic data"""
        try:
            analysis = {}
            
            if query_type == "salinity_temperature":
                # Analyze salinity-temperature relationship
                correlation = self.df[['temperature', 'salinity']].corr().iloc[0, 1]
                temp_stats = self.df['temperature'].describe()
                sal_stats = self.df['salinity'].describe()
                
                analysis = {
                    'correlation': correlation,
                    'temperature_stats': temp_stats.to_dict(),
                    'salinity_stats': sal_stats.to_dict(),
                    'type': 'salinity_temperature'
                }
            
            elif query_type == "ocean_warming":
                # Analyze temperature trends over time
                self.df['date_dt'] = pd.to_datetime(self.df['date'])
                yearly_temps = self.df.groupby(self.df['date_dt'].dt.year)['temperature'].mean()
                
                if len(yearly_temps) > 1:
                    temp_trend = np.polyfit(yearly_temps.index, yearly_temps.values, 1)[0]
                else:
                    temp_trend = 0
                
                analysis = {
                    'yearly_temperatures': yearly_temps.to_dict(),
                    'warming_trend': temp_trend,
                    'avg_temperature': self.df['temperature'].mean(),
                    'type': 'ocean_warming'
                }
            
            elif query_type == "cyclone_prediction":
                # Analyze conditions relevant to cyclone formation
                surface_data = self.df[self.df['depth'] <= 50]  # Surface layer
                warm_water = surface_data[surface_data['temperature'] >= 26.5]  # Cyclone threshold
                
                analysis = {
                    'warm_water_percentage': len(warm_water) / len(surface_data) * 100,
                    'avg_surface_temp': surface_data['temperature'].mean(),
                    'high_temp_locations': warm_water[['latitude', 'longitude', 'temperature']].to_dict('records')[:10],
                    'type': 'cyclone_prediction'
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {'error': str(e), 'type': query_type}
    
    def generate_intelligent_response(self, question: str, context_data: List[Dict] = None, 
                                    web_results: List[Dict] = None, analysis_data: Dict = None) -> str:
        """Generate intelligent response using AI model"""
        
        # Prepare context
        context_parts = []
        
        if context_data:
            context_parts.append("ARGO Data Context:")
            for i, data in enumerate(context_data[:3]):
                context_parts.append(f"Record {i+1}: Float {data['float_id']} on {data['date']} - "
                                   f"Temp: {data['temperature']}°C, Salinity: {data['salinity']} PSU, "
                                   f"Depth: {data['depth']}m, Location: {data['latitude']}, {data['longitude']}")
        
        if analysis_data and 'error' not in analysis_data:
            context_parts.append("\nData Analysis:")
            if analysis_data['type'] == 'salinity_temperature':
                context_parts.append(f"Temperature-Salinity correlation: {analysis_data['correlation']:.3f}")
                context_parts.append(f"Average temperature: {analysis_data['temperature_stats']['mean']:.2f}°C")
                context_parts.append(f"Average salinity: {analysis_data['salinity_stats']['mean']:.2f} PSU")
            elif analysis_data['type'] == 'ocean_warming':
                context_parts.append(f"Ocean warming trend: {analysis_data['warming_trend']:.4f}°C per year")
                context_parts.append(f"Average temperature: {analysis_data['avg_temperature']:.2f}°C")
            elif analysis_data['type'] == 'cyclone_prediction':
                context_parts.append(f"Warm water (>26.5°C) percentage: {analysis_data['warm_water_percentage']:.1f}%")
                context_parts.append(f"Average surface temperature: {analysis_data['avg_surface_temp']:.2f}°C")
        
        if web_results:
            context_parts.append("\nWeb Search Results:")
            for result in web_results[:2]:
                context_parts.append(f"- {result['title']}: {result['snippet'][:100]}...")
        
        context = "\n".join(context_parts)
        
        # Create comprehensive prompt
        prompt = f"""You are an expert oceanographer and marine scientist analyzing ARGO float data. 
Provide a comprehensive, scientific answer to the following question using the provided context.

Question: {question}

Available Context:
{context}

Please provide a detailed, scientific explanation that:
1. Directly answers the question
2. Uses specific data from the ARGO dataset when available
3. Explains the oceanographic concepts involved
4. Discusses scientific implications
5. Is accessible but scientifically accurate

Response:"""
        
        # Generate AI response
        ai_response = self.ai_model.generate_response(prompt, max_tokens=800)
        
        # Fallback if AI model fails
        if "AI model not available" in ai_response or "Error generating" in ai_response:
            return self._generate_fallback_response(question, context_data, analysis_data, web_results)
        
        return ai_response
    
    def _generate_fallback_response(self, question: str, context_data: List[Dict] = None,
                                   analysis_data: Dict = None, web_results: List[Dict] = None) -> str:
        """Generate fallback response when AI model is unavailable"""
        
        question_lower = question.lower()
        
        if "salinity" in question_lower and "temperature" in question_lower:
            if analysis_data and analysis_data['type'] == 'salinity_temperature':
                corr = analysis_data['correlation']
                return f"""Based on the ARGO dataset analysis:

**Salinity-Temperature Relationship:**
- Correlation coefficient: {corr:.3f}
- Average temperature: {analysis_data['temperature_stats']['mean']:.2f}°C
- Average salinity: {analysis_data['salinity_stats']['mean']:.2f} PSU

The correlation of {corr:.3f} indicates a {'strong positive' if corr > 0.7 else 'moderate positive' if corr > 0.3 else 'weak'} relationship between salinity and temperature in this dataset. In general oceanography, this relationship is influenced by evaporation (increases both), precipitation (decreases salinity), and mixing processes."""
        
        elif "warming" in question_lower or "trend" in question_lower:
            if analysis_data and analysis_data['type'] == 'ocean_warming':
                trend = analysis_data['warming_trend']
                return f"""Based on the temporal analysis of ARGO data:

**Ocean Warming Trends:**
- Temperature trend: {trend:.4f}°C per year
- Average temperature: {analysis_data['avg_temperature']:.2f}°C
- Data shows {'warming' if trend > 0 else 'cooling' if trend < 0 else 'stable'} trend

This trend analysis helps understand regional ocean temperature changes, which are crucial for climate monitoring and marine ecosystem health."""
        
        elif "cyclone" in question_lower:
            if analysis_data and analysis_data['type'] == 'cyclone_prediction':
                warm_pct = analysis_data['warm_water_percentage']
                return f"""Based on ARGO data analysis for cyclone-relevant conditions:

**Cyclone Formation Indicators:**
- Warm water (>26.5°C) coverage: {warm_pct:.1f}%
- Average surface temperature: {analysis_data['avg_surface_temp']:.2f}°C

ARGO floats provide critical data for cyclone prediction by monitoring:
1. Sea surface temperatures (cyclones need >26.5°C)
2. Ocean heat content in upper layers
3. Salinity patterns affecting water density
4. Temperature profiles indicating mixed layer depth

Higher percentages of warm water indicate more favorable conditions for tropical cyclone development."""
        
        else:
            # Generic response with available data
            response_parts = ["Based on the available ARGO float data:\n"]
            
            if context_data:
                response_parts.append("Recent measurements show:")
                for data in context_data[:3]:
                    response_parts.append(f"• Float {data['float_id']}: {data['temperature']}°C, {data['salinity']} PSU at {data['depth']}m depth")
            
            if web_results:
                response_parts.append(f"\nAdditional information: {web_results[0]['snippet'][:200]}...")
            
            return "\n".join(response_parts)
    
    def query_database(self, float_id: str = None, date: str = None, 
                      lat_range: Tuple[float, float] = None,
                      lon_range: Tuple[float, float] = None,
                      temp_range: Tuple[float, float] = None,
                      limit: int = 100) -> List[ARGORecord]:
        """Enhanced database query with more parameters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM argo_data WHERE 1=1"
            params = []
            
            if float_id:
                query += " AND float_id = ?"
                params.append(float_id)
            
            if date:
                query += " AND date = ?"
                params.append(date)
            
            if lat_range:
                query += " AND latitude BETWEEN ? AND ?"
                params.extend(lat_range)
            
            if lon_range:
                query += " AND longitude BETWEEN ? AND ?"
                params.extend(lon_range)
            
            if temp_range:
                query += " AND temperature BETWEEN ? AND ?"
                params.extend(temp_range)
            
            query += f" ORDER BY date DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            records = []
            for row in results:
                records.append(ARGORecord(
                    float_id=row[1], date=row[2], latitude=row[3], longitude=row[4],
                    depth=row[5], temperature=row[6], salinity=row[7], pressure=row[8]
                ))
            
            return records
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []
    
    def analyze_query(self, question: str) -> Dict[str, Any]:
        """Enhanced query analysis"""
        question_lower = question.lower()
        
        # Extract float ID
        float_match = re.search(r'float\s+(\w+)', question_lower)
        float_id = None
        if float_match:
            float_num = float_match.group(1)
            if not float_num.startswith('argo_'):
                float_id = f"ARGO_{float_num.zfill(4)}"
            else:
                float_id = float_num.upper()
        
        # Extract date
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
        date = date_match.group(1) if date_match else None
        
        # Extract coordinates
        coords = self.parse_coordinates(question)
        
        # Determine query type and analysis needed
        needs_web_search = False
        needs_analysis = None
        
        if ("salinity" in question_lower and "temperature" in question_lower and 
            "relationship" in question_lower):
            query_type = "salinity_temperature_analysis"
            needs_analysis = "salinity_temperature"
        elif ("warming" in question_lower or "trend" in question_lower):
            query_type = "ocean_warming_analysis"
            needs_analysis = "ocean_warming"
            needs_web_search = True
        elif ("cyclone" in question_lower or "hurricane" in question_lower or "typhoon" in question_lower):
            query_type = "cyclone_analysis"
            needs_analysis = "cyclone_prediction"
            needs_web_search = True
        elif any(term in question_lower for term in ["climate", "global warming", "sea level", "coral"]):
            query_type = "climate_analysis"
            needs_web_search = True
        elif "temperature" in question_lower:
            query_type = "temperature"
        elif "salinity" in question_lower:
            query_type = "salinity"
        elif "depth" in question_lower:
            query_type = "depth"
        elif "pressure" in question_lower:
            query_type = "pressure"
        elif "satellite" in question_lower or "image" in question_lower:
            query_type = "satellite"
        else:
            query_type = "general"
            if any(term in question_lower for term in ["explain", "what", "how", "why"]):
                needs_web_search = True
        
        return {
            'type': query_type,
            'float_id': float_id,
            'date': date,
            'coordinates': coords,
            'needs_web_search': needs_web_search,
            'needs_analysis': needs_analysis
        }
    
    def parse_coordinates(self, text: str) -> Optional[Tuple[float, float]]:
        """Parse coordinates from text"""
        patterns = [
            r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',
            r'lat\s+(-?\d+\.?\d*)\s+lon\s+(-?\d+\.?\d*)',
            r'latitude\s+(-?\d+\.?\d*)\s+longitude\s+(-?\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return lat, lon
                except ValueError:
                    continue
        return None
    
    def get_satellite_image(self, lat: float, lon: float, date: str = None) -> str:
        """Get satellite image URL"""
        try:
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return "Invalid coordinates. Latitude must be -90 to 90, longitude -180 to 180."
            
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            
            zoom = 2
            tile_x = int((lon + 180) / 360 * (2 ** zoom))
            tile_y = int((90 - lat) / 180 * (2 ** zoom))
            
            url = f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/{date}/250m/{zoom}/{tile_y}/{tile_x}.jpg"
            return f"Satellite image for {lat}, {lon} on {date}: {url}"
            
        except Exception as e:
            return f"Error generating satellite image: {e}"
    
    def chat(self, question: str) -> str:
        """Enhanced main chat interface with AI and web search"""
        try:
            logger.info(f"Processing: {question[:50]}...")
            
            # Analyze the query
            query_info = self.analyze_query(question)
            
            # Handle satellite imagery
            if query_info['type'] == 'satellite':
                if query_info['coordinates']:
                    lat, lon = query_info['coordinates']
                    return self.get_satellite_image(lat, lon, query_info['date'])
                else:
                    return "Please provide coordinates for satellite imagery (e.g., 'show satellite image at 20.5, -80.3')."
            
            # Prepare data context
            context_data = []
            analysis_data = None
            web_results = []
            
            # Get relevant ARGO data
            records = self.query_database(
                float_id=query_info['float_id'],
                date=query_info['date'],
                limit=50
            )
            
            if records:
                context_data = [
                    {
                        'float_id': r.float_id,
                        'date': r.date,
                        'latitude': r.latitude,
                        'longitude': r.longitude,
                        'depth': r.depth,
                        'temperature': r.temperature,
                        'salinity': r.salinity,
                        'pressure': r.pressure
                    } for r in records[:10]
                ]
            
            # Perform analysis if needed
            if query_info['needs_analysis']:
                analysis_data = self.analyze_ocean_patterns(query_info['needs_analysis'])
            
            # Perform web search if needed
            if query_info['needs_web_search']:
                search_query = question
                if query_info['type'] == 'ocean_warming_analysis':
                    search_query = "ocean warming climate change ARGO float data"
                elif query_info['type'] == 'cyclone_analysis':
                    search_query = "ARGO float data cyclone hurricane prediction ocean temperature"
                elif query_info['type'] == 'climate_analysis':
                    search_query = f"oceanography climate change {question}"
                
                web_results = self.web_search.search(search_query, max_results=3)
            
            # Generate intelligent response
            if (context_data or analysis_data or web_results or 
                query_info['type'] in ['salinity_temperature_analysis', 'ocean_warming_analysis', 'cyclone_analysis']):
                
                return self.generate_intelligent_response(
                    question, context_data, web_results, analysis_data
                )
            
            # Fallback to simple data search if no specific data found
            if self.search_engine:
                search_results = self.search_engine.search(question, k=5)
                if search_results:
                    best_match = search_results[0][1]  # metadata from best match
                    context_data = [best_match]
                    
                    return self.generate_intelligent_response(
                        question, context_data, web_results, analysis_data
                    )
            
            # Final fallback
            return ("I couldn't find specific ARGO data for your query. Try asking about:\n"
                   "• Temperature-salinity relationships\n"
                   "• Ocean warming trends\n"
                   "• Cyclone prediction using ocean data\n"
                   "• Specific float measurements (e.g., 'float 1023 data')")
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error processing question: {e}"
    
    def get_data_summary(self) -> str:
        """Get enhanced dataset summary with analysis"""
        if self.df is None:
            return "No data loaded"
        
        # Calculate additional statistics
        temp_by_depth = self.df.groupby(pd.cut(self.df['depth'], bins=5))['temperature'].mean()
        seasonal_temps = self.df.groupby(pd.to_datetime(self.df['date']).dt.month)['temperature'].mean()
        
        return f"""🌊 Enhanced ARGO Dataset Summary:
📊 Dataset Overview:
• Total records: {len(self.df):,}
• Unique floats: {self.df['float_id'].nunique()}
• Date range: {self.df['date'].min()} to {self.df['date'].max()}
• Geographic coverage: {self.df['latitude'].min():.1f}° to {self.df['latitude'].max():.1f}° lat, {self.df['longitude'].min():.1f}° to {self.df['longitude'].max():.1f}° lon

🌡️ Temperature Analysis:
• Range: {self.df['temperature'].min():.1f}°C to {self.df['temperature'].max():.1f}°C
• Average: {self.df['temperature'].mean():.1f}°C
• Standard deviation: {self.df['temperature'].std():.2f}°C

🧂 Salinity Analysis:
• Range: {self.df['salinity'].min():.1f} to {self.df['salinity'].max():.1f} PSU
• Average: {self.df['salinity'].mean():.1f} PSU

🏊 Depth Coverage:
• Range: {self.df['depth'].min():.1f}m to {self.df['depth'].max():.1f}m
• Average measurement depth: {self.df['depth'].mean():.1f}m

🤖 AI Capabilities:
• Phi-3.5 model: {'✅ Loaded' if self.ai_model.model else '❌ Not available'}
• Web search: ✅ Enabled
• Pattern analysis: ✅ Enabled"""

def main():
    """Enhanced main function with AI capabilities"""
    print("🌊 Enhanced ARGO AI Chatbot with Phi-3.5 & Web Search")
    print("=" * 60)
    
    try:
        bot = EnhancedARGOChatbot()
        
        print("✅ Enhanced ARGO AI Chatbot is ready!")
        print(bot.get_data_summary())
        print(f"\n🔍 Example intelligent queries:")
        print("• 'Explain the relationship between salinity and temperature in ARGO float data'")
        print("• 'What trends in ocean warming can be observed from this dataset?'")
        print("• 'How can ARGO float data be useful for predicting cyclones?'")
        print("• 'What is the impact of climate change on ocean salinity patterns?'")
        print("• 'Show me temperature data for float 1023'")
        print("• 'What are the latest developments in ARGO float technology?'")
        print("\nCommands: 'help', 'summary', 'quit'\n")
        
        while True:
            try:
                user_input = input("🤖 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                    
                if user_input.lower() in ["help", "h"]:
                    print("""
🔍 Available query types:
• 🧠 Analytical questions: "Explain the relationship between salinity and temperature"
• 📈 Trend analysis: "What ocean warming trends do you see?"
• 🌀 Climate predictions: "How does ARGO data help predict cyclones?"
• 📊 Specific data: "Temperature of float 1023 on 2022-05-12"
• 🛰️ Satellite imagery: "Satellite image at 25.5, -80.2"
• 🌐 General knowledge: "What are ARGO floats used for?"
• 📋 Dataset info: "summary" or "stats"

The chatbot uses AI to provide detailed scientific explanations!
                    """)
                    continue
                
                if user_input.lower() in ["summary", "stats", "info"]:
                    print(f"📊 {bot.get_data_summary()}\n")
                    continue
                
                print("🤔 Thinking...")
                start_time = time.time()
                answer = bot.chat(user_input)
                response_time = time.time() - start_time
                
                print(f"🌊 {answer}")
                print(f"⏱️ Response time: {response_time:.2f} seconds\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"❌ Error: {e}")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        print("\n🔧 Setup Requirements:")
        print("1. Install dependencies: pip install llama-cpp-python pandas numpy requests")
        print("2. Download Phi-3.5-mini-instruct-Q4_K_M.gguf to ./models/ directory")
        print("3. Ensure internet connection for web search functionality")

if __name__ == "__main__":
    main()