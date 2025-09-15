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
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import torch 
import uuid


# Suppress warnings
warnings.filterwarnings("ignore")

# Try to import llama-cpp-python for Phi-3.5 model
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")

# Try to import LangChain and FAISS for RAG functionality
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not installed. Install with: pip install langchain langchain-community faiss-cpu sentence-transformers")

# GPU device detection (MPS for macOS, CUDA for others)
import torch

def get_device():
    """Detect and return the best available device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()
print(f"Using device: {DEVICE}")

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
    """Data class for ARGO measurements - Updated for real dataset"""
    n_points: int
    cycle_number: int
    data_mode: str
    direction: str
    platform_number: str
    position_qc: str
    pressure: float
    pres_error: float
    pres_qc: str
    salinity: float
    psal_error: float
    psal_qc: str
    temperature: float
    temp_error: float
    temp_qc: str
    time_qc: str
    latitude: float
    longitude: float
    date: str
    region: str

class RAGRetriever:
    """RAG (Retrieval Augmented Generation) system using LangChain and FAISS with GPU support"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.vector_store = None
        self.embeddings = None
        self.device = DEVICE
        self.faiss_index_path = self.data_dir / "argo_index.faiss"
        self.metadata_path = self.data_dir / "argo_metadata.pkl"
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_rag()
        else:
            logger.warning("LangChain not available. RAG functionality disabled.")
    
    def _initialize_rag(self):
        """Initialize RAG system with embeddings and vector store using GPU when available"""
        try:
            # Initialize embeddings model with GPU support
            model_kwargs = {'device': self.device} if self.device != 'cpu' else {'device': 'cpu'}
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"Embeddings model initialized on {self.device}")
            
            # Try to load existing FAISS index
            if self.faiss_index_path.exists() and self.metadata_path.exists():
                logger.info("Loading existing FAISS index...")
                self.vector_store = FAISS.load_local(
                    str(self.data_dir), 
                    self.embeddings,
                    index_name="argo_index"
                )
                logger.info("FAISS index loaded successfully")
            else:
                logger.info("FAISS index not found. RAG retrieval will be unavailable until index is created.")
                
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            self.vector_store = None
    
    def create_vector_store(self, documents: List[str], metadata: List[Dict]):
        """Create and save FAISS vector store from documents using GPU acceleration"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. Cannot create vector store.")
            return False
            
        try:
            # Convert to LangChain Document objects
            docs = []
            for i, (doc_text, doc_metadata) in enumerate(zip(documents, metadata)):
                docs.append(Document(
                    page_content=doc_text,
                    metadata=doc_metadata
                ))
            
            # Create vector store with GPU acceleration
            logger.info(f"Creating FAISS vector store with {len(docs)} documents on {self.device}...")
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            
            # Save the vector store
            self.vector_store.save_local(str(self.data_dir), index_name="argo_index")
            logger.info(f"FAISS vector store created and saved successfully on {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Tuple[str, Dict]]:
        """Retrieve relevant documents using RAG with GPU acceleration"""
        if not self.vector_store:
            logger.warning("Vector store not available. Using fallback search.")
            return []
        
        try:
            # Perform similarity search with GPU acceleration
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            retrieved_docs = []
            for doc, score in results:
                retrieved_docs.append((doc.page_content, doc.metadata))
            
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error during RAG retrieval: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if RAG system is available"""
        return LANGCHAIN_AVAILABLE and self.vector_store is not None

class PlottingService:
    """Enhanced service for generating oceanographic plots with image saving"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        self.figsize = (12, 8)
        self.dpi = 100
        
        # Create static directory for saving images
        self.static_dir = Path("static")
        self.static_dir.mkdir(exist_ok=True)
        
    def _save_and_encode_plot(self, fig, plot_type: str) -> dict:
        """Save plot as PNG and return both file path and base64 encoded data"""
        try:
            # Generate unique filename
            filename = f"{plot_type}_{uuid.uuid4().hex[:8]}.png"
            filepath = self.static_dir / filename
            
            # Save as PNG file
            fig.savefig(filepath, format='png', bbox_inches='tight', dpi=self.dpi, 
                       facecolor='white', edgecolor='none')
            
            # Also create base64 encoded version for immediate display
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=self.dpi,
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close(fig)
            
            return {
                'type': 'image',
                'filename': filename,
                'filepath': str(filepath),
                'base64': img_base64,
                'url': f"/static/{filename}"
            }
            
        except Exception as e:
            plt.close(fig)
            raise e
            
    def create_temp_vs_depth_plot(self, df: pd.DataFrame, title: str = "Temperature vs Depth") -> dict:
        """Create temperature vs depth plot and return image data"""
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Filter out invalid data
            valid_data = df[(df['temperature'].notna()) & (df['pressure'].notna())]
            
            if len(valid_data) == 0:
                plt.close(fig)
                return {
                    'type': 'error',
                    'message': 'No valid temperature/depth data available for plotting.'
                }
            
            # Limit data points to 200 to reduce clutter
            if len(valid_data) > 200:
                valid_data = valid_data.sample(n=200, random_state=42)
            
            # Create scatter plot with color based on temperature
            scatter = ax.scatter(valid_data['temperature'], valid_data['pressure'], 
                               c=valid_data['temperature'], cmap='coolwarm', alpha=0.7, s=30)
            
            ax.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Pressure/Depth (dbar)', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.invert_yaxis()  # Depth increases downward
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold')
            
            # Add grid and styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            # Add data point count annotation
            ax.text(0.02, 0.98, f'Data points: {len(valid_data)}', 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            result = self._save_and_encode_plot(fig, 'temp_vs_depth')
            result['data_points'] = len(valid_data)
            result['description'] = f"Temperature vs Depth plot with {len(valid_data)} data points"
            
            return result
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Error creating temperature vs depth plot: {str(e)}"
            }
            
    def create_salinity_vs_temp_plot(self, df: pd.DataFrame, title: str = "Salinity vs Temperature") -> dict:
        """Create salinity vs temperature plot and return image data"""
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Filter out invalid data
            valid_data = df[(df['temperature'].notna()) & (df['salinity'].notna())]
            
            if len(valid_data) == 0:
                plt.close(fig)
                return {
                    'type': 'error',
                    'message': 'No valid salinity/temperature data available for plotting.'
                }
            
            # Limit data points to 200 to reduce clutter
            if len(valid_data) > 200:
                valid_data = valid_data.sample(n=200, random_state=42)
            
            # Create scatter plot with color based on depth (pressure)
            scatter = ax.scatter(valid_data['temperature'], valid_data['salinity'], 
                               c=valid_data['pressure'] if 'pressure' in valid_data.columns else 'blue', 
                               cmap='viridis', alpha=0.7, s=30)
            
            ax.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Salinity (PSU)', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add colorbar if pressure data available
            if 'pressure' in valid_data.columns and valid_data['pressure'].notna().any():
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Pressure/Depth (dbar)', fontsize=12, fontweight='bold')
            
            # Add grid and styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            # Add data point count annotation
            ax.text(0.02, 0.98, f'Data points: {len(valid_data)}', 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            result = self._save_and_encode_plot(fig, 'salinity_vs_temp')
            result['data_points'] = len(valid_data)
            result['description'] = f"Salinity vs Temperature plot with {len(valid_data)} data points"
            
            return result
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Error creating salinity vs temperature plot: {str(e)}"
            }
            
    def create_salinity_vs_depth_plot(self, df: pd.DataFrame, title: str = "Salinity vs Depth") -> dict:
        """Create salinity vs depth plot and return image data"""
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Filter out invalid data
            valid_data = df[(df['salinity'].notna()) & (df['pressure'].notna())]
            
            if len(valid_data) == 0:
                plt.close(fig)
                return {
                    'type': 'error',
                    'message': 'No valid salinity/depth data available for plotting.'
                }
            
            # Limit data points to 200 to reduce clutter
            if len(valid_data) > 200:
                valid_data = valid_data.sample(n=200, random_state=42)
            
            # Create scatter plot with color based on salinity
            scatter = ax.scatter(valid_data['salinity'], valid_data['pressure'], 
                               c=valid_data['salinity'], cmap='plasma', alpha=0.7, s=30)
            
            ax.set_xlabel('Salinity (PSU)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Pressure/Depth (dbar)', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.invert_yaxis()  # Depth increases downward
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Salinity (PSU)', fontsize=12, fontweight='bold')
            
            # Add grid and styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            # Add data point count annotation
            ax.text(0.02, 0.98, f'Data points: {len(valid_data)}', 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            result = self._save_and_encode_plot(fig, 'salinity_vs_depth')
            result['data_points'] = len(valid_data)
            result['description'] = f"Salinity vs Depth plot with {len(valid_data)} data points"
            
            return result
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Error creating salinity vs depth plot: {str(e)}"
            }
            
    def create_pressure_vs_time_plot(self, df: pd.DataFrame, title: str = "Pressure vs Time") -> dict:
        """Create pressure vs time plot"""
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Filter out invalid data
            valid_data = df[(df['pressure'].notna()) & (df['date'].notna())]
            
            if len(valid_data) == 0:
                plt.close(fig)
                return {
                    'type': 'error',
                    'message': 'No valid pressure/time data available for plotting.'
                }
            
            # Convert date to datetime
            valid_data['date_dt'] = pd.to_datetime(valid_data['date'])
            valid_data = valid_data.sort_values('date_dt')
            
            # Limit data points to 200 to reduce clutter
            if len(valid_data) > 200:
                valid_data = valid_data.sample(n=200, random_state=42).sort_values('date_dt')
            
            # Create line plot
            ax.plot(valid_data['date_dt'], valid_data['pressure'], 'o-', alpha=0.7, markersize=4)
            
            ax.set_xlabel('Date', fontsize=14, fontweight='bold')
            ax.set_ylabel('Pressure (dbar)', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add grid and styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Add data point count annotation
            ax.text(0.02, 0.98, f'Data points: {len(valid_data)}', 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            result = self._save_and_encode_plot(fig, 'pressure_vs_time')
            result['data_points'] = len(valid_data)
            result['description'] = f"Pressure vs Time plot with {len(valid_data)} data points"
            
            return result
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Error creating pressure vs time plot: {str(e)}"
            }
class DuckDuckGoSearch:
    """Simple DuckDuckGo search integration"""
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
        self.instant_answer_url = "https://api.duckduckgo.com/"
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search DuckDuckGo and return results"""
        try:
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
            
            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractText', 'DuckDuckGo Result'),
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', ''),
                    'source': 'DuckDuckGo Instant Answer'
                })
            
            for topic in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('FirstURL', {}).get('Text', 'Related Topic'),
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', {}).get('FirstURL', ''),
                        'source': 'DuckDuckGo Related'
                    })
            
            if not results:
                results.append({
                    'title': f'Search results for: {query}',
                    'snippet': f'I searched for "{query}" but couldn\'t find specific instant answers.',
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
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_idx, score in sorted_docs:
            results.append((self.documents[doc_idx], self.metadata[doc_idx]))
        
        return results

class Phi35Model:
    """Phi-3.5-mini-instruct model wrapper with GPU support"""
    
    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self.device = DEVICE
        self.load_model()
    
    def load_model(self):
        """Load the Phi-3.5 model with GPU support"""
        try:
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python not available")
            
            model_file = Path(self.model_path) / "Phi-3.5-mini-instruct-Q4_K_M.gguf"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            # Configure GPU settings
            n_gpu_layers = -1 if self.device in ['cuda', 'mps'] else 0
            
            self.model = Llama(
                model_path=str(model_file),
                n_ctx=4096,
                n_threads=4,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            logger.info(f"Phi-3.5 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Phi-3.5 model: {e}")
            self.model = None
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using Phi-3.5 model"""
        if self.model is None:
            return "AI model not available. Using fallback response generation."
        
        try:
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
    """Enhanced ARGO Chatbot with real dataset, GPU support, and plotting capabilities"""
    
    def __init__(self, data_dir: str = ".", models_dir: str = "./models"):
        """Initialize Enhanced ARGO Chatbot with real dataset"""
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.df = None
        self.db_path = self.data_dir / "argo_data.db"
        self.csv_path = self.data_dir / "argo_data.csv"  # Real dataset
        self.search_engine = None
        self.web_search = DuckDuckGoSearch()
        self.ai_model = Phi35Model(self.models_dir)
        self.rag_retriever = RAGRetriever(self.data_dir)
        self.plotter = PlottingService()
        
        logger.info(f"Starting Enhanced ARGO Chatbot with real dataset on {DEVICE}...")
        self._load_real_data()
        self._setup_database()
        self._setup_search()
        self._setup_rag()
        logger.info("Enhanced ARGO Chatbot with real dataset initialized successfully")
    
    def _load_real_data(self):
        """Load real ARGO dataset"""
        try:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Real ARGO dataset not found: {self.csv_path}")
            
            logger.info("Loading real ARGO dataset...")
            self.df = pd.read_csv(self.csv_path)
            
            # Ensure date column is properly formatted
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            
            # Clean numeric columns
            numeric_columns = ['pressure', 'salinity', 'temperature', 'latitude', 'longitude']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            logger.info(f"Loaded {len(self.df)} records from real ARGO dataset")
            logger.info(f"Dataset columns: {list(self.df.columns)}")
            
        except Exception as e:
            logger.error(f"Error loading real ARGO data: {e}")
            raise
    
    def _setup_database(self):
        """Setup SQLite database for real dataset"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table matching real dataset structure
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS argo_data (
                    id INTEGER PRIMARY KEY,
                    n_points INTEGER,
                    cycle_number INTEGER,
                    data_mode TEXT,
                    direction TEXT,
                    platform_number TEXT,
                    position_qc TEXT,
                    pressure REAL,
                    pres_error REAL,
                    pres_qc TEXT,
                    salinity REAL,
                    psal_error REAL,
                    psal_qc TEXT,
                    temperature REAL,
                    temp_error REAL,
                    temp_qc TEXT,
                    time_qc TEXT,
                    latitude REAL,
                    longitude REAL,
                    date TEXT,
                    region TEXT
                )
            ''')
            
            # Create indices for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_platform ON argo_data(platform_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON argo_data(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_location ON argo_data(latitude, longitude)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_region ON argo_data(region)')
            
            # Check if data exists
            cursor.execute('SELECT COUNT(*) FROM argo_data')
            count = cursor.fetchone()[0]
            
            if count == 0:
                # Insert real data
                columns = list(self.df.columns)
                placeholders = ','.join(['?' for _ in columns])
                
                for _, row in self.df.iterrows():
                    cursor.execute(f'''
                        INSERT INTO argo_data ({','.join(columns)})
                        VALUES ({placeholders})
                    ''', tuple(row))
                
                conn.commit()
                logger.info(f"Inserted {len(self.df)} records into database")
            else:
                logger.info(f"Database already contains {count} records")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def _setup_search(self):
        """Setup search with real dataset content"""
        try:
            documents = []
            metadata = []
            
            for _, row in self.df.iterrows():
                # Create rich content for real dataset
                content = f"""
ARGO Platform {row.get('platform_number', 'Unknown')} oceanographic measurement
Cycle: {row.get('cycle_number', 'N/A')}, Data Mode: {row.get('data_mode', 'N/A')}
Date: {row.get('date', 'N/A')}, Region: {row.get('region', 'Unknown')}
Geographic location: latitude {row.get('latitude', 'N/A')} degrees, longitude {row.get('longitude', 'N/A')} degrees
Oceanographic measurements: pressure {row.get('pressure', 'N/A')} dbar, temperature {row.get('temperature', 'N/A')} degrees Celsius, salinity {row.get('salinity', 'N/A')} PSU
Quality control: Position QC {row.get('position_qc', 'N/A')}, Pressure QC {row.get('pres_qc', 'N/A')}, Temperature QC {row.get('temp_qc', 'N/A')}, Salinity QC {row.get('psal_qc', 'N/A')}
Scientific context: ARGO float oceanographic profile data for climate research and ocean monitoring
Keywords: oceanography marine science CTD profile temperature salinity pressure ocean circulation climate
                """.strip()
                
                documents.append(content)
                metadata.append({k: v for k, v in row.items() if pd.notna(v)})
            
            self.search_engine = SimpleTextSearch(documents, metadata)
            logger.info(f"Initialized search with {len(documents)} real dataset documents")
            
        except Exception as e:
            logger.error(f"Error setting up search: {e}")
            raise
    
    def _setup_rag(self):
        """Setup RAG system with real dataset"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. RAG functionality disabled.")
            return
        
        if not self.rag_retriever.is_available() and self.df is not None:
            logger.info("Creating RAG vector store from real ARGO data...")
            
            documents = []
            metadata = []
            
            for _, row in self.df.iterrows():
                # Enhanced document content for better RAG
                content = f"""
ARGO Float Scientific Measurement Record
Platform Number: {row.get('platform_number', 'Unknown')}
Measurement Cycle: {row.get('cycle_number', 'N/A')}
Date: {row.get('date', 'N/A')}
Ocean Region: {row.get('region', 'Unknown')}
Geographic Coordinates: Latitude {row.get('latitude', 'N/A')}° N/S, Longitude {row.get('longitude', 'N/A')}° E/W
Hydrographic Profile Data:
- Water Pressure: {row.get('pressure', 'N/A')} decibars (measurement depth indicator)
- Sea Water Temperature: {row.get('temperature', 'N/A')} degrees Celsius
- Practical Salinity: {row.get('salinity', 'N/A')} PSU (Practical Salinity Units)
Data Quality Indicators: Position QC: {row.get('position_qc', 'N/A')}, Temperature QC: {row.get('temp_qc', 'N/A')}, Salinity QC: {row.get('psal_qc', 'N/A')}
Measurement Errors: Temperature error ±{row.get('temp_error', 'N/A')}°C, Salinity error ±{row.get('psal_error', 'N/A')} PSU, Pressure error ±{row.get('pres_error', 'N/A')} dbar
Scientific Applications: Ocean circulation studies, climate monitoring, thermohaline circulation analysis, marine ecosystem research
Data Mode: {row.get('data_mode', 'N/A')} (Real-time or Delayed-mode quality controlled data)
Research Keywords: ARGO autonomous float, oceanography, hydrographic survey, CTD profile, ocean temperature, ocean salinity, sea water density, ocean pressure, marine science, climate research
                """.strip()
                
                documents.append(content)
                metadata.append({
                    'platform_number': row.get('platform_number', 'Unknown'),
                    'cycle_number': row.get('cycle_number'),
                    'date': row.get('date'),
                    'region': row.get('region', 'Unknown'),
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude'),
                    'pressure': row.get('pressure'),
                    'temperature': row.get('temperature'),
                    'salinity': row.get('salinity'),
                    'data_mode': row.get('data_mode'),
                    'doc_type': 'argo_real_measurement'
                })
            
            success = self.rag_retriever.create_vector_store(documents, metadata)
            if success:
                logger.info("RAG vector store created successfully from real dataset")
            else:
                logger.warning("Failed to create RAG vector store")
    
    def detect_plot_request(self, question: str) -> Optional[str]:
        """Enhanced plot detection with more patterns"""
        question_lower = question.lower()
    
        plot_keywords = {
            # Temperature vs Depth patterns
            r'(temp.*depth|depth.*temp|temperature.*depth|depth.*temperature|temp.*vs.*depth|depth.*vs.*temp)': 'temp_depth',
        
            # Salinity vs Temperature patterns  
            r'(sal.*temp|temp.*sal|salinity.*temperature|temperature.*salinity|sal.*vs.*temp|temp.*vs.*sal)': 'sal_temp',
            
            # Salinity vs Depth patterns
            r'(sal.*depth|depth.*sal|salinity.*depth|depth.*salinity|sal.*vs.*depth|depth.*vs.*sal)': 'sal_depth',
            
            # Pressure vs Time patterns
            r'(pressure.*time|time.*pressure|pressure.*vs.*time|time.*vs.*pressure)': 'pressure_time',
            
            # General plotting keywords
            r'(plot|graph|chart|visualize|show.*graph|show.*plot|display.*graph)': 'general_plot'
        }
        
        # Check for plot request indicators
        plot_indicators = ['plot', 'graph', 'chart', 'visualize', 'show', 'display', 'draw']
        has_plot_indicator = any(indicator in question_lower for indicator in plot_indicators)
    
        if has_plot_indicator:
            for pattern, plot_type in plot_keywords.items():
                if plot_type != 'general_plot' and re.search(pattern, question_lower):
                    return plot_type
            
            # If general plot request, try to infer from context
            if 'temperature' in question_lower and 'depth' in question_lower:
                return 'temp_depth'
            elif 'salinity' in question_lower and 'temperature' in question_lower:
                return 'sal_temp'
            elif 'salinity' in question_lower and 'depth' in question_lower:
                return 'sal_depth'
            elif 'pressure' in question_lower and 'time' in question_lower:
                return 'pressure_time'
        
        return None
    
    def generate_plot(self, plot_type: str, filters: Dict = None) -> dict:
        """Generate requested plot and return image data"""
        try:
            logger.info(f"Generating {plot_type} plot...")
            
            # Apply filters if provided
            data = self.df.copy()
            if filters:
                for key, value in filters.items():
                    if key in data.columns and pd.notna(value):
                        if isinstance(value, str):
                            data = data[data[key].str.contains(value, case=False, na=False)]
                        else:
                            data = data[data[key] == value]
        
            # Generate appropriate plot
            if plot_type == 'temp_depth':
                return self.plotter.create_temp_vs_depth_plot(data, "Temperature vs Depth Profile")
            elif plot_type == 'sal_temp':
                return self.plotter.create_salinity_vs_temp_plot(data, "Salinity vs Temperature Relationship")
            elif plot_type == 'sal_depth':
                return self.plotter.create_salinity_vs_depth_plot(data, "Salinity vs Depth Profile")
            elif plot_type == 'pressure_time':
                return self.plotter.create_pressure_vs_time_plot(data, "Pressure vs Time Series")
            else:
                return {
                    'type': 'error',
                    'message': f"Unknown plot type: {plot_type}"
                }
            
        except Exception as e:
            logger.error(f"Error generating {plot_type} plot: {e}")
            return {
                'type': 'error',
                'message': f"Error generating plot: {str(e)}"
            }
    def analyze_ocean_patterns(self, query_type: str) -> Dict[str, Any]:
        """Analyze patterns in real oceanographic data"""
        try:
            analysis = {}
            
            if query_type == "salinity_temperature":
                # Analyze salinity-temperature relationship
                valid_data = self.df[(self.df['temperature'].notna()) & (self.df['salinity'].notna())]
                if len(valid_data) > 0:
                    correlation = valid_data[['temperature', 'salinity']].corr().iloc[0, 1]
                    temp_stats = valid_data['temperature'].describe()
                    sal_stats = valid_data['salinity'].describe()
                    
                    analysis = {
                        'correlation': correlation,
                        'temperature_stats': temp_stats.to_dict(),
                        'salinity_stats': sal_stats.to_dict(),
                        'data_points': len(valid_data),
                        'type': 'salinity_temperature'
                    }
                else:
                    analysis = {'error': 'No valid temperature/salinity data', 'type': query_type}
            
            elif query_type == "ocean_warming":
                # Analyze temperature trends over time
                valid_data = self.df[(self.df['temperature'].notna()) & (self.df['date'].notna())]
                if len(valid_data) > 0:
                    valid_data['date_dt'] = pd.to_datetime(valid_data['date'])
                    yearly_temps = valid_data.groupby(valid_data['date_dt'].dt.year)['temperature'].mean()
                    
                    if len(yearly_temps) > 1:
                        temp_trend = np.polyfit(yearly_temps.index, yearly_temps.values, 1)[0]
                    else:
                        temp_trend = 0
                    
                    analysis = {
                        'yearly_temperatures': yearly_temps.to_dict(),
                        'warming_trend': temp_trend,
                        'avg_temperature': valid_data['temperature'].mean(),
                        'data_points': len(valid_data),
                        'type': 'ocean_warming'
                    }
                else:
                    analysis = {'error': 'No valid temperature/date data', 'type': query_type}
            
            elif query_type == "cyclone_prediction":
                # Analyze conditions relevant to cyclone formation
                surface_data = self.df[(self.df['pressure'] <= 50) & (self.df['temperature'].notna())]
                if len(surface_data) > 0:
                    warm_water = surface_data[surface_data['temperature'] >= 26.5]
                    
                    analysis = {
                        'warm_water_percentage': len(warm_water) / len(surface_data) * 100,
                        'avg_surface_temp': surface_data['temperature'].mean(),
                        'surface_data_points': len(surface_data),
                        'warm_water_points': len(warm_water),
                        'type': 'cyclone_prediction'
                    }
                else:
                    analysis = {'error': 'No valid surface temperature data', 'type': query_type}
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {'error': str(e), 'type': query_type}
    
    def generate_intelligent_response(self, question: str, context_data: List[Dict] = None, 
                                    web_results: List[Dict] = None, analysis_data: Dict = None,
                                    rag_results: List[Tuple[str, Dict]] = None) -> str:
        """Generate intelligent response using AI model with RAG enhancement"""
        
        # Prepare context
        context_parts = []
        
        if rag_results:
            context_parts.append("RAG Retrieved Context:")
            for i, (doc_content, doc_metadata) in enumerate(rag_results[:2]):
                if 'platform_number' in doc_metadata:
                    context_parts.append(f"RAG Document {i+1}: Platform {doc_metadata['platform_number']} - {doc_content[:300]}...")
        
        if context_data:
            context_parts.append("ARGO Data Context:")
            for i, data in enumerate(context_data[:3]):
                platform = data.get('platform_number', 'Unknown')
                temp = data.get('temperature', 'N/A')
                sal = data.get('salinity', 'N/A')
                pressure = data.get('pressure', 'N/A')
                context_parts.append(f"Record {i+1}: Platform {platform} - Temp: {temp}°C, Salinity: {sal} PSU, Pressure: {pressure} dbar")
        
        if analysis_data and 'error' not in analysis_data:
            context_parts.append("\nData Analysis:")
            if analysis_data['type'] == 'salinity_temperature':
                context_parts.append(f"Temperature-Salinity correlation: {analysis_data['correlation']:.3f}")
                context_parts.append(f"Data points: {analysis_data['data_points']}")
                context_parts.append(f"Average temperature: {analysis_data['temperature_stats']['mean']:.2f}°C")
                context_parts.append(f"Average salinity: {analysis_data['salinity_stats']['mean']:.2f} PSU")
            elif analysis_data['type'] == 'ocean_warming':
                context_parts.append(f"Ocean warming trend: {analysis_data['warming_trend']:.4f}°C per year")
                context_parts.append(f"Average temperature: {analysis_data['avg_temperature']:.2f}°C")
                context_parts.append(f"Data points: {analysis_data['data_points']}")
            elif analysis_data['type'] == 'cyclone_prediction':
                context_parts.append(f"Warm water (>26.5°C) percentage: {analysis_data['warm_water_percentage']:.1f}%")
                context_parts.append(f"Surface data points: {analysis_data['surface_data_points']}")
        
        if web_results:
            context_parts.append("\nWeb Search Results:")
            for result in web_results[:2]:
                context_parts.append(f"- {result['title']}: {result['snippet'][:100]}...")
        
        context = "\n".join(context_parts)
        
        # Create comprehensive prompt with RAG enhancement
        prompt = f"""You are an expert oceanographer analyzing real ARGO float data with enhanced retrieval capabilities. 
Provide a comprehensive, scientific answer using the provided context.

Question: {question}

Available Context (including RAG-retrieved information):
{context}

Please provide a detailed, scientific explanation that:
1. Directly answers the question
2. Uses specific data from the real ARGO dataset
3. Incorporates RAG-retrieved information when available
4. Explains oceanographic concepts
5. Discusses scientific implications
6. Maintains scientific accuracy

Response:"""
        
        # Generate AI response
        ai_response = self.ai_model.generate_response(prompt, max_tokens=800)
        
        # Fallback if AI model fails
        if "AI model not available" in ai_response or "Error generating" in ai_response:
            return self._generate_fallback_response(question, context_data, analysis_data, web_results, rag_results)
        
        return ai_response
    
    def _generate_fallback_response(self, question: str, context_data: List[Dict] = None,
                                   analysis_data: Dict = None, web_results: List[Dict] = None,
                                   rag_results: List[Tuple[str, Dict]] = None) -> str:
        """Generate fallback response when AI model is unavailable"""
        
        question_lower = question.lower()
        
        # Include RAG information in fallback
        rag_info = ""
        if rag_results:
            for doc_content, doc_metadata in rag_results[:1]:
                if 'platform_number' in doc_metadata:
                    rag_info += f"\nRAG Context: Platform {doc_metadata['platform_number']} shows temperature {doc_metadata.get('temperature', 'N/A')}°C and salinity {doc_metadata.get('salinity', 'N/A')} PSU."
        
        if "salinity" in question_lower and "temperature" in question_lower:
            if analysis_data and analysis_data['type'] == 'salinity_temperature' and 'error' not in analysis_data:
                corr = analysis_data['correlation']
                return f"""Based on real ARGO dataset analysis:

**Salinity-Temperature Relationship:**
- Correlation coefficient: {corr:.3f}
- Data points analyzed: {analysis_data['data_points']}
- Average temperature: {analysis_data['temperature_stats']['mean']:.2f}°C
- Average salinity: {analysis_data['salinity_stats']['mean']:.2f} PSU

The correlation indicates a {'strong positive' if corr > 0.7 else 'moderate positive' if corr > 0.3 else 'weak'} relationship between salinity and temperature in this real dataset.{rag_info}"""
        
        elif "warming" in question_lower or "trend" in question_lower:
            if analysis_data and analysis_data['type'] == 'ocean_warming' and 'error' not in analysis_data:
                trend = analysis_data['warming_trend']
                return f"""Based on real ARGO temporal analysis:

**Ocean Temperature Trends:**
- Temperature trend: {trend:.4f}°C per year
- Average temperature: {analysis_data['avg_temperature']:.2f}°C
- Data points: {analysis_data['data_points']}
- Trend shows {'warming' if trend > 0 else 'cooling' if trend < 0 else 'stable'} pattern

This analysis from real ARGO data provides insights into regional temperature changes.{rag_info}"""
        
        elif "cyclone" in question_lower:
            if analysis_data and analysis_data['type'] == 'cyclone_prediction' and 'error' not in analysis_data:
                warm_pct = analysis_data['warm_water_percentage']
                return f"""Based on real ARGO data for cyclone analysis:

**Cyclone Formation Conditions:**
- Warm water (>26.5°C) coverage: {warm_pct:.1f}%
- Surface data points: {analysis_data['surface_data_points']}
- Warm water locations: {analysis_data['warm_water_points']}

Real ARGO data provides critical cyclone prediction capabilities through ocean temperature monitoring.{rag_info}"""
        
        else:
            response_parts = ["Based on real ARGO float dataset:\n"]
            
            if rag_results:
                response_parts.append("RAG-Enhanced Context:")
                for doc_content, doc_metadata in rag_results[:2]:
                    if 'platform_number' in doc_metadata:
                        temp = doc_metadata.get('temperature', 'N/A')
                        sal = doc_metadata.get('salinity', 'N/A')
                        response_parts.append(f"• Platform {doc_metadata['platform_number']}: {temp}°C, {sal} PSU")
            
            if context_data:
                response_parts.append("Current measurements:")
                for data in context_data[:3]:
                    platform = data.get('platform_number', 'Unknown')
                    temp = data.get('temperature', 'N/A')
                    sal = data.get('salinity', 'N/A')
                    response_parts.append(f"• Platform {platform}: {temp}°C, {sal} PSU")
            
            if web_results:
                response_parts.append(f"\nAdditional information: {web_results[0]['snippet'][:200]}...")
            
            return "\n".join(response_parts)
    
    def query_database(self, platform_number: str = None, date: str = None, 
                      region: str = None, lat_range: Tuple[float, float] = None,
                      lon_range: Tuple[float, float] = None,
                      temp_range: Tuple[float, float] = None,
                      limit: int = 100) -> List[ARGORecord]:
        """Enhanced database query for real dataset"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM argo_data WHERE 1=1"
            params = []
            
            if platform_number:
                query += " AND platform_number = ?"
                params.append(platform_number)
            
            if date:
                query += " AND date = ?"
                params.append(date)
            
            if region:
                query += " AND region = ?"
                params.append(region)
            
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
                    n_points=row[1] or 0, cycle_number=row[2] or 0, data_mode=row[3] or '',
                    direction=row[4] or '', platform_number=row[5] or '', position_qc=row[6] or '',
                    pressure=row[7] or 0.0, pres_error=row[8] or 0.0, pres_qc=row[9] or '',
                    salinity=row[10] or 0.0, psal_error=row[11] or 0.0, psal_qc=row[12] or '',
                    temperature=row[13] or 0.0, temp_error=row[14] or 0.0, temp_qc=row[15] or '',
                    time_qc=row[16] or '', latitude=row[17] or 0.0, longitude=row[18] or 0.0,
                    date=row[19] or '', region=row[20] or ''
                ))
            
            return records
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []
    
    def analyze_query(self, question: str) -> Dict[str, Any]:
        """Enhanced query analysis with better plot detection"""
        question_lower = question.lower()
        
        # Extract platform number
        platform_match = re.search(r'platform\s+(\w+)', question_lower)
        platform_number = platform_match.group(1) if platform_match else None
        
        # Extract date
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
        date = date_match.group(1) if date_match else None
        
        # Extract region
        region_match = re.search(r'region\s+(\w+)', question_lower)
        region = region_match.group(1) if region_match else None
        
        # Extract coordinates
        coords = self.parse_coordinates(question)
        
        # Check for plot requests - this is the key enhancement
        plot_request = self.detect_plot_request(question)
        
        # Determine query type and analysis needed
        needs_web_search = False
        needs_analysis = None
        needs_rag = True
        
        if plot_request:
            query_type = f"plot_{plot_request}"
            # For plot requests, we might not need web search unless it's for context
            needs_web_search = False
        elif ("salinity" in question_lower and "temperature" in question_lower and 
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
        elif "pressure" in question_lower:
            query_type = "pressure"
        elif "satellite" in question_lower or "image" in question_lower:
            query_type = "satellite"
            needs_rag = False
        else:
            query_type = "general"
            if any(term in question_lower for term in ["explain", "what", "how", "why"]):
                needs_web_search = True
    
        return {
        'type': query_type,
        'platform_number': platform_number,
        'date': date,
        'region': region,
        'coordinates': coords,
        'plot_request': plot_request,
        'needs_web_search': needs_web_search,
        'needs_analysis': needs_analysis,
        'needs_rag': needs_rag
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
        """Enhanced chat interface with plotting, real dataset, and GPU-accelerated RAG"""
        try:
            logger.info(f"Processing: {question[:50]}...")
            
            # Analyze the query
            query_info = self.analyze_query(question)
            
            # Handle plot requests
            if query_info['plot_request']:
                filters = {}
                if query_info['platform_number']:
                    filters['platform_number'] = query_info['platform_number']
                if query_info['region']:
                    filters['region'] = query_info['region']
                if query_info['date']:
                    filters['date'] = query_info['date']
                
                plot_result = self.generate_plot(query_info['plot_request'], filters)
                return plot_result  # Return dict for image response
            
            # Handle satellite imagery
            if query_info['type'] == 'satellite':
                if query_info['coordinates']:
                    lat, lon = query_info['coordinates']
                    return self.get_satellite_image(lat, lon, query_info['date'])
                else:
                    return "Please provide coordinates for satellite imagery (e.g., 'show satellite image at 20.5, -80.3')."
            
            # ... rest of the existing chat method remains the same ...
            # (Continue with existing logic for text responses)
            
            # Prepare data context
            context_data = []
            analysis_data = None
            web_results = []
            rag_results = []
            
            # Perform RAG retrieval if needed and available
            if query_info['needs_rag'] and self.rag_retriever.is_available():
                logger.info("Performing RAG retrieval...")
                rag_results = self.rag_retriever.retrieve_relevant_docs(question, k=5)
                logger.info(f"RAG retrieved {len(rag_results)} relevant documents")
            
            # Get relevant ARGO data
            records = self.query_database(
                platform_number=query_info['platform_number'],
                date=query_info['date'],
                region=query_info['region'],
                limit=50
            )
            
            if records:
                context_data = [
                    {
                        'platform_number': r.platform_number,
                        'cycle_number': r.cycle_number,
                        'date': r.date,
                        'region': r.region,
                        'latitude': r.latitude,
                        'longitude': r.longitude,
                        'pressure': r.pressure,
                        'temperature': r.temperature,
                        'salinity': r.salinity,
                        'data_mode': r.data_mode
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
            
            # Generate intelligent response with all enhancements
            if (context_data or analysis_data or web_results or rag_results or 
                query_info['type'] in ['salinity_temperature_analysis', 'ocean_warming_analysis', 'cyclone_analysis']):
                
                return self.generate_intelligent_response(
                    question, context_data, web_results, analysis_data, rag_results
                )
            
            # Fallback to search
            if self.search_engine:
                search_results = self.search_engine.search(question, k=5)
                if search_results:
                    best_match = search_results[0][1]
                    context_data = [best_match]
                    
                    return self.generate_intelligent_response(
                        question, context_data, web_results, analysis_data, rag_results
                    )
            
            # Final fallback
            return ("I couldn't find specific data for your query. Try asking about:\n"
                   "• Temperature-salinity relationships\n"
                   "• Ocean warming trends\n"
                   "• Cyclone prediction using ocean data\n"
                   "• Plotting: 'plot temperature vs depth' or 'show salinity vs temperature graph'\n"
                   "• Specific platform data (e.g., 'platform 6902746 data')")
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error processing question: {e}"
    def get_data_summary(self) -> str:
        """Get enhanced dataset summary with real data statistics"""
        if self.df is None:
            return "No data loaded"
        
        # Calculate statistics for real dataset
        total_platforms = self.df['platform_number'].nunique() if 'platform_number' in self.df.columns else 0
        date_range = f"{self.df['date'].min()} to {self.df['date'].max()}" if 'date' in self.df.columns else "Unknown"
        regions = self.df['region'].nunique() if 'region' in self.df.columns else 0
        
        # Temperature statistics
        temp_stats = self.df['temperature'].describe() if 'temperature' in self.df.columns else None
        sal_stats = self.df['salinity'].describe() if 'salinity' in self.df.columns else None
        pressure_stats = self.df['pressure'].describe() if 'pressure' in self.df.columns else None
        
        rag_status = "Active" if self.rag_retriever.is_available() else "Not available"
        langchain_status = "Available" if LANGCHAIN_AVAILABLE else "Not installed"
        
        return f"""Real ARGO Dataset Summary with GPU-Accelerated RAG:
Dataset Overview:
• Total records: {len(self.df):,}
• Unique platforms: {total_platforms}
• Regions covered: {regions}
• Date range: {date_range}
• Device: {DEVICE}

Temperature Analysis:
• Range: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C
• Average: {temp_stats['mean']:.1f}°C
• Valid measurements: {temp_stats['count']:,.0f}

Salinity Analysis:
• Range: {sal_stats['min']:.1f} to {sal_stats['max']:.1f} PSU
• Average: {sal_stats['mean']:.1f} PSU
• Valid measurements: {sal_stats['count']:,.0f}

Pressure Coverage:
• Range: {pressure_stats['min']:.1f} to {pressure_stats['max']:.1f} dbar
• Average: {pressure_stats['mean']:.1f} dbar

AI Capabilities:
• Phi-3.5 model: {'Loaded' if self.ai_model.model else 'Not available'} ({DEVICE})
• Web search: Enabled
• Pattern analysis: Enabled
• Plotting service: Available

RAG System Status:
• LangChain: {langchain_status}
• FAISS Vector Store: {rag_status}
• GPU Acceleration: {'Yes' if DEVICE in ['cuda', 'mps'] else 'No'} ({DEVICE})
• Enhanced context retrieval: {'Available' if self.rag_retriever.is_available() else 'Basic search only'}"""

def main():
    """Main function with real dataset and GPU support"""
    print(f"Enhanced ARGO AI Chatbot with Real Dataset & GPU Support ({DEVICE})")
    print("=" * 80)
    
    try:
        bot = EnhancedARGOChatbot()
        
        print("Enhanced ARGO AI Chatbot with real dataset is ready!")
        print(bot.get_data_summary())
        print(f"\nExample queries with real dataset:")
        print("• 'Explain the relationship between salinity and temperature'")
        print("• 'Plot temperature vs depth for Indian Ocean data'")
        print("• 'Show salinity vs temperature graph'")
        print("• 'What are ocean warming trends in this dataset?'")
        print("• 'Platform 6902746 temperature data'")
        print("• 'How does cyclone prediction use ARGO data?'")
        print("\nCommands: 'help', 'summary', 'quit'\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                    
                if user_input.lower() in ["help", "h"]:
                    print(f"""
Available features with real dataset:
• Analytical questions: "Relationship between salinity and temperature"
• Plotting: "plot temperature vs depth", "graph salinity vs temperature"
• Trend analysis: "Ocean warming trends"
• Platform-specific: "Platform 6902746 data"
• Regional analysis: "Indian Ocean temperature patterns"
• Climate predictions: "Cyclone prediction using ARGO data"
• General knowledge: "What are ARGO floats?"

GPU Status: {DEVICE}
RAG Enhancement: {'Active' if bot.rag_retriever.is_available() else 'Not available'}
Plotting: Available (matplotlib + seaborn)
                    """)
                    continue
                
                if user_input.lower() in ["summary", "stats", "info"]:
                    print(f"{bot.get_data_summary()}\n")
                    continue
                
                print("Processing... (with GPU-accelerated RAG)")
                start_time = time.time()
                answer = bot.chat(user_input)
                response_time = time.time() - start_time
                
                print(f"{answer}")
                print(f"Response time: {response_time:.2f} seconds\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"Error: {e}")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        print("\nSetup Requirements:")
        print("1. pip install llama-cpp-python pandas numpy requests matplotlib seaborn")
        print("2. pip install langchain langchain-community faiss-cpu sentence-transformers torch")
        print("3. Place argo_data.csv in the current directory")
        print("4. Download Phi-3.5-mini-instruct-Q4_K_M.gguf to ./models/ directory")
        print("5. Ensure internet connection for web search")

if __name__ == "__main__":
    main()
