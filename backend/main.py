# --- Core Libraries ---
import os
import pandas as pd
import re 
from uuid import UUID, uuid4
from datetime import datetime, date 
from typing import List, Annotated, Optional

# --- Web Framework & Data Validation ---
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Date/Time Parsing ---
from dotenv import load_dotenv
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

# --- Machine Learning & AI ---
from sentence_transformers import SentenceTransformer, util
import torch

# --- Database & Authentication (Google Firebase) ---
import firebase_admin
from firebase_admin import credentials, auth, firestore

# Load environment variables from a .env file (if it exists)
# Useful for storing sensitive information like API keys without hardcoding them.
load_dotenv()

# --- Firebase Admin SDK Initialization ---
# This block sets up the connection to the Firebase project, which is used for
# user authentication and the Firestore database.

# Initialize a global variable for the Firestore database client.
db = None
try:
    # Define the path to the Firebase service account key file.
    SERVICE_ACCOUNT_PATH = "firebase-service-account.json"
    # Check if the required service account key file exists before proceeding.
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise FileNotFoundError(f"Service account key not found at {SERVICE_ACCOUNT_PATH}")
    
    # Load the credentials from the service account file.
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    
    # Initialize the Firebase Admin SDK app, but only if it hasn't been initialized already.
    # This prevents errors if the module is reloaded.
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    else:
        print("Firebase Admin SDK already initialized.")
        
    # Get a client instance for the Firestore service. This object will be used
    # for all database operations (reading/writing user profiles).
    db = firestore.client()
    print("Firestore client obtained successfully.")
except Exception as e:
    # If any part of the Firebase setup fails, log a critical error.
    # The application will likely be non-functional without this connection.
    print(f"CRITICAL ERROR initializing Firebase Admin SDK or Firestore client: {e}")


# --- Global Variables for Job Data and ML Model ---
# These variables are loaded once when the application starts to avoid
# reloading data and models on every API request, which would be very slow.

# DataFrame to hold all the job postings data.
jobs_df: Optional[pd.DataFrame] = None
JOBS_DATA_PATH = "data/jobs.csv" # The path to the job dataset CSV file.

# The name of the Sentence Transformer model to use for creating semantic embeddings.
MODEL_NAME = 'all-mpnet-base-v2' 
# Variable to hold the loaded Sentence Transformer model object.
sentence_model: Optional[SentenceTransformer] = None
# Variable to hold the pre-computed tensor of embeddings for all jobs.
job_embeddings: Optional[torch.Tensor] = None

# Create a file-safe slug from the model name for caching purposes.
model_slug = MODEL_NAME.replace('/', '_').replace('-', '_')
EMBEDDINGS_CACHE_DIR = "data/cache"
# Define the full path for the cached embeddings file. This ensures that if the
# model or dataset changes, a new cache file will be used.
EMBEDDINGS_CACHE_PATH = os.path.join(EMBEDDINGS_CACHE_DIR, f"job_embeddings_{model_slug}_job_opportunities_dataset_v_refined_text.pt") 


# --- Data Loading and Preprocessing Function ---

def load_and_clean_job_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Loads job data from a CSV file, cleans it, and prepares it for the matching algorithm.
    This includes cleaning text, parsing structured data like experience, and
    creating a combined text field for semantic embedding.
    
    Args:
        csv_path: The file path to the jobs CSV.
        
    Returns:
        A cleaned and prepared pandas DataFrame, or None if loading fails.
    """
    try:
        print(f"Attempting to load new job dataset from: {csv_path}...")
        if not os.path.exists(csv_path):
            print(f"ERROR: Job dataset file not found at {csv_path}.")
            return None
            
        temp_df = pd.read_csv(csv_path)
        print(f"Initial rows loaded from new CSV: {len(temp_df)}")

        # Standardize column names to be lowercase with underscores for easier access.
        temp_df.columns = [str(col).strip().lower().replace(' ', '_') for col in temp_df.columns]
        
        # Fill any missing values in key text columns with empty strings to prevent errors.
        temp_df['job_description'] = temp_df['job_description'].fillna('').astype(str)
        temp_df['role'] = temp_df['role'].fillna('N/A').astype(str) 
        temp_df['job_title'] = temp_df['job_title'].fillna('N/A').astype(str) 
        temp_df['company'] = temp_df['company'].fillna('N/A').astype(str)
        temp_df['experience'] = temp_df['experience'].fillna('Unknown').astype(str)
        # The 'skills' column from the CSV is expected to contain full sentences.
        temp_df['skills_text_from_csv'] = temp_df.get('skills', pd.Series([''] * len(temp_df))).fillna('').astype(str) 

        # --- Experience Level Parsing ---
        def parse_experience_to_level(exp_str: str) -> str:
            """Converts a raw experience string (e.g., "3-5 years") into a standardized level."""
            if pd.isna(exp_str) or exp_str.lower() in ['n/a', 'unknown', '']: return "Unknown"
            # Find all numbers in the string.
            numbers = re.findall(r'\d+', exp_str)
            if not numbers:
                # If no numbers are found, check for keywords like "entry", "senior", etc.
                exp_str_lower = exp_str.lower()
                if "entry" in exp_str_lower or "fresher" in exp_str_lower: return "Entry-Level"
                if "junior" in exp_str_lower: return "Junior"
                if "senior" in exp_str_lower: return "Senior"
                if "lead" in exp_str_lower or "manager" in exp_str_lower: return "Senior"
                return "Unknown"
            try:
                # Use the first number found to categorize the experience level.
                years = int(numbers[0])
                if years < 1: return "Entry-Level"
                elif years < 3: return "Junior"
                elif years < 7: return "Mid-Level"
                else: return "Senior"
            except: return "Unknown"
        # Apply this function to the 'experience' column to create a new standardized column.
        temp_df['parsed_experience_level'] = temp_df['experience'].apply(parse_experience_to_level)
        
        # --- Keyword Extraction ---
        # A predefined list of common tech skills to look for.
        COMMON_TECH_KEYWORDS = ['java', 'python', 'sql', 'excel', 'aws', 'azure', 'cisco', 
                                'cybersecurity', 'pmp', 'agile', 'r', 'docker', 'jenkins', 
                                'react', 'angular', 'vue', 'node.js', 'javascript', 'typescript',
                                'html', 'css', 'tensorflow', 'pytorch', 'machine learning', 'data analysis']
        def extract_keywords_from_sentences(text_data: str) -> List[str]:
            """Extracts predefined keywords from a block of text."""
            if pd.isna(text_data) or not isinstance(text_data, str): return []
            found_skills = set()
            text_data_lower = text_data.lower()
            for keyword in COMMON_TECH_KEYWORDS:
                # Use regex with word boundaries (\b) to match whole words only.
                # This prevents matching 'r' in 'architecture', for example.
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_data_lower):
                    found_skills.add(keyword.capitalize()) # Add the skill in a consistent format.
            return list(found_skills)
        # Apply this function to the raw skills text to get a list of keywords for each job.
        temp_df['parsed_job_keywords'] = temp_df['skills_text_from_csv'].apply(extract_keywords_from_sentences)

        # --- Column Renaming and Final Selection ---
        # Rename columns from the CSV to match the Pydantic 'Job' model for consistency.
        column_mappings = {
            'job_id': 'id', 'role': 'title', 'job_description': 'description',
            'company': 'company',
            'parsed_job_keywords': 'requiredSkills',
            'parsed_experience_level': 'experience_level_required'
        }
        temp_df.rename(columns={k: v for k, v in column_mappings.items() if k in temp_df.columns}, inplace=True)
        
        # Generate a unique ID for each job if one doesn't already exist.
        if 'id' not in temp_df.columns:
            temp_df['id'] = [uuid4().hex for _ in range(len(temp_df))]
        else:
            temp_df['id'] = temp_df['id'].astype(str)
        
        # Ensure all required columns for the API exist, adding default values if they are missing.
        for field in ['title', 'company',  'description', 'requiredSkills', 'experience_level_required']:
            if field not in temp_df.columns:
                default_value = [] if field == 'requiredSkills' else 'N/A'
                temp_df[field] = default_value
        
        # Select only the columns that are needed for the application.
        required_model_fields = ['id', 'title', 'company', 'description', 'requiredSkills', 'experience_level_required']
        final_columns = [col for col in required_model_fields if col in temp_df.columns]
        cleaned_df = temp_df[final_columns].copy()
        
        # Drop any rows that are missing an 'id' or 'title' as they are essential.
        cleaned_df.dropna(subset=['id', 'title'], inplace=True)

        if cleaned_df.empty:
            print("WARNING: DataFrame is empty after cleaning/filtering.")
            return None
        
        # --- Text Preparation for Embedding ---
        # Create a single, rich text field for each job by combining its title,
        # the original skills sentences, and the job description. This provides
        # the best context for the semantic model.
        cleaned_df['title_for_emb'] = cleaned_df['title'].fillna('').astype(str)
        cleaned_df['skills_sentences_for_emb'] = temp_df.get('skills_text_from_csv', pd.Series([''] * len(cleaned_df))).fillna('').astype(str)
        cleaned_df['description_for_emb'] = cleaned_df['description'].fillna('').astype(str)
        
        cleaned_df['text_for_embedding'] = cleaned_df['title_for_emb'] + ". " + \
                                           cleaned_df['skills_sentences_for_emb'] + ". " + \
                                           cleaned_df['description_for_emb']
        # Clean up the combined text to remove redundant punctuation and whitespace.
        cleaned_df['text_for_embedding'] = cleaned_df['text_for_embedding'].str.replace(r'\.\s*\.', '.', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
        print(f"Example text for JOB embedding: {cleaned_df['text_for_embedding'].iloc[0][:300] + '...' if not cleaned_df.empty and cleaned_df['text_for_embedding'].iloc[0] else 'N/A'}")
        
        print(f"Successfully cleaned and prepared {len(cleaned_df)} IT job postings.")
        return cleaned_df

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ERROR: Failed to load or process IT job dataset from CSV: {e}")
        return None


# --- Model and Embeddings Loading ---

# Load the Sentence Transformer model from HuggingFace.
try:
    print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    sentence_model = SentenceTransformer(MODEL_NAME)
    print(f"Sentence Transformer model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Sentence Transformer model '{MODEL_NAME}': {e}")
    sentence_model = None

# This block handles loading the job embeddings, using a cache to speed up startup.
if sentence_model:
    # Ensure the cache directory exists.
    try:
        os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create cache directory '{EMBEDDINGS_CACHE_DIR}': {e}")

    # --- Step 1: Try to load embeddings from the cache file. ---
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        print(f"Attempting to load cached job embeddings from {EMBEDDINGS_CACHE_PATH}...")
        try:
            job_embeddings = torch.load(EMBEDDINGS_CACHE_PATH)
            print(f"Successfully loaded job embeddings from cache. Shape: {job_embeddings.shape}")
            # Also load the job data DataFrame.
            jobs_df = load_and_clean_job_data(JOBS_DATA_PATH)
            # **Crucial check**: Ensure the loaded embeddings match the number of jobs in the current CSV.
            # If they don't match, the cache is stale and must be recomputed.
            if jobs_df is None or len(jobs_df) != job_embeddings.shape[0]:
                print(f"WARNING: Mismatch or error loading DataFrame with cache. Recomputing embeddings.")
                job_embeddings = None; jobs_df = None 
        except Exception as e:
            print(f"ERROR loading embeddings from cache or verifying DataFrame: {e}. Recomputing.")
            job_embeddings = None; jobs_df = None
            
    # --- Step 2: If cache loading failed or was skipped, compute the embeddings. ---
    if job_embeddings is None: 
        print(f"Recomputing job embeddings for {MODEL_NAME}...")
        # Load the job data if it's not already loaded.
        if jobs_df is None: jobs_df = load_and_clean_job_data(JOBS_DATA_PATH)
        
        if jobs_df is not None and not jobs_df.empty:
            # Check if the required text column exists for embedding.
            if 'text_for_embedding' not in jobs_df.columns or jobs_df['text_for_embedding'].isnull().all():
                print("ERROR: 'text_for_embedding' column is missing or all null in jobs_df. Cannot compute embeddings.")
                job_embeddings = None
            else:
                # Convert the text column to a list for the model.
                job_texts_for_embedding = jobs_df['text_for_embedding'].fillna('').tolist()
                try:
                    # This is the core ML step: encode all job texts into vectors.
                    # 'convert_to_tensor=True' is important for performance with PyTorch.
                    job_embeddings = sentence_model.encode(job_texts_for_embedding, convert_to_tensor=True, show_progress_bar=True)
                    print(f"Job embeddings calculated. Shape: {job_embeddings.shape}. Saving to {EMBEDDINGS_CACHE_PATH}...")
                    # Save the newly computed embeddings tensor to the cache file for next time.
                    torch.save(job_embeddings, EMBEDDINGS_CACHE_PATH)
                    print("Job embeddings saved to cache.")
                except Exception as e:
                    print(f"ERROR calculating or saving embeddings: {e}"); job_embeddings = None
        else:
            print("Jobs DataFrame empty/not loaded; cannot compute embeddings."); job_embeddings = None
else:
    # If the model itself failed to load, semantic matching is impossible.
    print("WARNING: Sentence Transformer model not loaded. Semantic matching will be unavailable.")
    if jobs_df is None: jobs_df = load_and_clean_job_data(JOBS_DATA_PATH)


# --- Pydantic Models for API Data Validation ---
# Pydantic models define the structure and data types of API requests and responses.
# This provides automatic validation, serialization, and documentation.

class SkillsUpdateRequest(BaseModel):
    skills: List[str] = Field(..., example=["Python", "SQL"])

class EducationItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    degree: str
    school: str
    startYear: Optional[int] = None
    endYear: Optional[int] = None

class EducationUpdateRequest(BaseModel):
    education: List[EducationItem]

class ExperienceItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    company: str
    startDate: Optional[str] = None # Using string for flexibility (e.g., "2020-01" or "Jan 2020")
    endDate: Optional[str] = None   # Can be a date or "Present"

class ExperienceUpdateRequest(BaseModel):
    experience: List[ExperienceItem]

# Defines the components that make up the final match score.
# This is useful for debugging and providing insights on the frontend.
class ScoreComponents(BaseModel):
    semantic_profile_score: float
    keyword_skill_score: float
    experience_level_score: float
    education_semantic_score: float

# The base model for a job listing.
class Job(BaseModel):
    id: str
    title: str
    company: Optional[str] = None
    description: Optional[str] = ""
    requiredSkills: List[str] = Field(default_factory=list)
    experience_level_required: Optional[str] = None 

# Extends the base Job model to include matching-specific information.
class MatchedJob(Job):
    matchScore: float = Field(..., example=0.85)
    score_components: ScoreComponents
    matching_keywords: List[str] = Field(default_factory=list)

# The model for a full user profile response.
class UserProfileResponse(BaseModel):
    uid: str
    email: Optional[str] = None
    displayName: Optional[str] = None
    photoURL: Optional[str] = None
    createdAt: Optional[datetime] = None
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    saved_job_ids: List[str] = Field(default_factory=list)


# --- FastAPI Application Setup ---
app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing) to allow the frontend
# (e.g., a React app running on localhost:3000) to communicate with this backend.
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"]  # Allow all headers
)


# --- Authentication Middleware (FastAPI Dependency) ---
async def get_current_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    """
    A FastAPI dependency that verifies the Firebase JWT token from the
    Authorization header. It protects endpoints by ensuring the user is logged in.
    
    Returns:
        The decoded token payload as a dictionary, containing user info like UID.
        
    Raises:
        HTTPException: If the header is missing, malformed, or the token is invalid.
    """
    # Check for the presence of the Authorization header.
    if authorization is None: 
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    
    # The header should be in the format "Bearer <token>".
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format")
    
    id_token = parts[1]
    
    # Verify the token using the Firebase Admin SDK.
    # `check_revoked=True` ensures that disabled users or logged-out sessions are rejected.
    try:
        return auth.verify_id_token(id_token, check_revoked=True)
    # Provide specific error messages for common authentication failures.
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    # Catch any other verification errors and return a generic server error.
    except Exception as e:
        print(f"Token verification failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not verify token")


# --- Scoring Helper Functions ---

def calculate_keyword_match_score(user_skills: List[str], job_skills: List[str]) -> float:
    """Calculates a keyword match score based on the Jaccard index."""
    if not job_skills: return 0.0 # Avoid division by zero if a job has no required skills.
    user_s = set(s.lower() for s in user_skills)
    job_s = set(s.lower() for s in job_skills)
    # The score is the ratio of matching skills to the total number of required skills.
    return len(user_s.intersection(job_s)) / len(job_s)

def get_user_total_experience_years(experience_list: List[dict]) -> float:
    """Calculates a user's total years of professional experience from their work history."""
    total_years = 0.0
    today = date.today()
    for exp_item_dict in experience_list:
        try:
            start_date_str = exp_item_dict.get('startDate')
            end_date_str = exp_item_dict.get('endDate')
            if not start_date_str: continue # Skip if there's no start date.
            
            # Use dateutil.parser for flexible date string parsing.
            start_date_obj = parse_date(start_date_str).date()
            
            # If the end date is missing or set to "Present", use today's date.
            end_date_obj = today
            if end_date_str and end_date_str.lower().strip() != 'present' and end_date_str.strip() != '':
                try:
                    end_date_obj = parse_date(end_date_str).date()
                except (ValueError, TypeError):
                    print(f"Could not parse end date: '{end_date_str}', assuming 'Present'.")
            
            # Handle cases where dates might be entered incorrectly.
            if end_date_obj < start_date_obj: 
                print(f"Warning: End date {end_date_obj} is before start date {start_date_obj} for experience: {exp_item_dict.get('title')}")
                continue
            
            # Use relativedelta to accurately calculate the time difference.
            delta = relativedelta(end_date_obj, start_date_obj)
            total_years += delta.years + (delta.months / 12.0) + (delta.days / 365.25)
        except Exception as e:
            print(f"Warning: Could not parse dates for experience item '{exp_item_dict.get('title')}': {e}")
    return total_years

def categorize_user_experience(total_years: float) -> str:
    """Categorizes a float number of years into a standardized experience level."""
    if total_years < 1: return "Entry-Level"
    elif total_years < 3: return "Junior"
    elif total_years < 7: return "Mid-Level"
    else: return "Senior"

def calculate_experience_level_match_score(user_level_str: str, job_level_str: Optional[str]) -> float:
    """
    Calculates a nuanced score based on how well the user's experience level
    matches the job's requirement. It rewards being overqualified less than a perfect match
    and penalizes being underqualified more severely.
    """
    # If the job has no specified level, return a neutral score.
    if not job_level_str or pd.isna(job_level_str) or job_level_str.lower().strip() in ['n/a', 'unknown', '']: return 0.5 
    
    # Normalize strings for comparison.
    user_level_norm = user_level_str.lower().replace('-', '').replace(' ', '')
    job_level_norm = job_level_str.lower().replace('-', '').replace(' ', '')
    
    # Map levels to numerical values for easier comparison.
    level_map = {"entrylevel": 0, "junior": 1, "midlevel": 2, "senior": 3}
    user_val = level_map.get(user_level_norm, -1)
    job_val = level_map.get(job_level_norm, -1)
    
    # If a level is not in our map, return a low score.
    if user_val == -1 or job_val == -1: return 0.25 
    
    # Perfect match.
    if user_val == job_val: return 1.0
    # User is overqualified. The score decreases the more overqualified they are.
    elif user_val > job_val:
        diff = user_val - job_val
        if diff == 1: return 0.7  # e.g., Mid-Level applying for Junior
        elif diff == 2: return 0.4  # e.g., Senior applying for Junior
        else: return 0.2
    # User is underqualified.
    elif job_val == user_val + 1: return 0.6 # e.g., Junior applying for Mid-Level (reasonable)
    elif job_val == user_val + 2: return 0.2 # e.g., Junior applying for Senior (less reasonable)
    else: return 0.05

def calculate_education_semantic_score(user_education_embeddings: Optional[torch.Tensor], job_text_embedding: torch.Tensor) -> float:
    """Calculates the max semantic similarity between any of the user's education items and the job."""
    if user_education_embeddings is None or user_education_embeddings.nelement() == 0 or job_text_embedding is None: return 0.0
    
    max_edu_score = 0.0
    try:
        # Ensure the job embedding is 2D for the similarity calculation.
        current_job_embedding_expanded = job_text_embedding
        if len(job_text_embedding.shape) == 1:
            current_job_embedding_expanded = job_text_embedding.unsqueeze(0)
            
        # Check for NaN values which can cause errors.
        if not (torch.isnan(user_education_embeddings).any() or torch.isnan(current_job_embedding_expanded).any()):
            # Calculate cosine similarity between all user education items and the current job.
            cosine_scores_tensor = util.cos_sim(user_education_embeddings, current_job_embedding_expanded)
            # The final education score is the highest similarity score from any single education item.
            if cosine_scores_tensor.numel() > 0:
                max_edu_score = float(torch.max(cosine_scores_tensor).item())
    except Exception as e:
        print(f"[WARN] Error calculating education semantic score: {e}")
    
    return max(0.0, max_edu_score) # Ensure score is not negative.


# --- API Endpoints ---

@app.get("/")
async def root():
    """A simple root endpoint to confirm that the API is running."""
    return {"message": "Job matching API is running"}

@app.get("/api/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user: Annotated[dict, Depends(get_current_user)]):
    """
    Fetches the current user's profile from Firestore.
    If a profile document doesn't exist for the user's UID, it creates a new one.
    """
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    
    user_uid = current_user.get("uid")
    
    try:
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()

        # If the user is accessing their profile for the first time, create their document in Firestore.
        if not user_doc.exists:
            user_info = auth.get_user(user_uid) # Get user info from Firebase Auth
            user_doc_ref.set({
                "email": user_info.email,
                "displayName": user_info.display_name,
                "photoURL": user_info.photo_url,
                "createdAt": datetime.fromtimestamp(user_info.user_metadata.creation_timestamp / 1000),
                "skills": [],
                "education": [],
                "experience": [],
                "saved_job_ids": []
            })
            print(f"Created Firestore profile for UID: {user_uid}")
            user_doc = user_doc_ref.get() # Re-fetch the newly created doc

        profile_data = user_doc.to_dict()
        profile_data["uid"] = user_uid
        # Ensure the saved_job_ids field exists for older profiles.
        profile_data.setdefault("saved_job_ids", [])

        # The dictionary is validated against the UserProfileResponse Pydantic model before returning.
        return UserProfileResponse(**profile_data)
    except Exception as e:
        print(f"Error fetching or creating profile for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch or create user profile")


# --- Saved Jobs Endpoints ---

@app.post("/api/profile/saved-jobs/{job_id}", status_code=status.HTTP_200_OK)
async def save_job(job_id: str, current_user: Annotated[dict, Depends(get_current_user)]):
    """Adds a job ID to the user's list of saved jobs in Firestore."""
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    
    user_uid = current_user.get("uid")
    user_doc_ref = db.collection("users").document(user_uid)
    try:
        # `firestore.ArrayUnion` is an atomic operation that adds an element to an array
        # only if it's not already present.
        user_doc_ref.update({"saved_job_ids": firestore.ArrayUnion([job_id])})
        return {"status": "success", "message": f"Job {job_id} saved."}
    except Exception as e:
        print(f"Error saving job {job_id} for user {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not save job.")

@app.delete("/api/profile/saved-jobs/{job_id}", status_code=status.HTTP_200_OK)
async def unsave_job(job_id: str, current_user: Annotated[dict, Depends(get_current_user)]):
    """Removes a job ID from the user's list of saved jobs."""
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    
    user_uid = current_user.get("uid")
    user_doc_ref = db.collection("users").document(user_uid)
    try:
        # `firestore.ArrayRemove` atomically removes all instances of a given element from an array.
        user_doc_ref.update({"saved_job_ids": firestore.ArrayRemove([job_id])})
        return {"status": "success", "message": f"Job {job_id} unsaved."}
    except Exception as e:
        print(f"Error unsaving job {job_id} for user {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not unsave job.")

@app.get("/api/profile/saved-jobs", response_model=List[Job])
async def get_saved_jobs(current_user: Annotated[dict, Depends(get_current_user)]):
    """Retrieves the full job details for all jobs the user has saved."""
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    if jobs_df is None: raise HTTPException(status_code=503, detail="Job data is not available")
    
    user_uid = current_user.get("uid")
    try:
        user_doc = db.collection("users").document(user_uid).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Get the list of saved job IDs from the user's profile.
        user_data = user_doc.to_dict()
        saved_ids = user_data.get("saved_job_ids", [])
        if not saved_ids:
            return [] 

        # Filter the main jobs DataFrame to get only the jobs whose IDs are in the saved list.
        saved_jobs_df = jobs_df[jobs_df['id'].isin(saved_ids)]
        saved_jobs_list = saved_jobs_df.to_dict(orient='records')
        
        # This step ensures the returned jobs are in the same order the user saved them.
        saved_jobs_dict = {job['id']: job for job in saved_jobs_list}
        ordered_saved_jobs = [saved_jobs_dict[job_id] for job_id in saved_ids if job_id in saved_jobs_dict]

        return ordered_saved_jobs
    except Exception as e:
        print(f"Error fetching saved jobs for user {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve saved jobs.")


# --- Profile Update Endpoints ---

@app.put("/api/profile/skills", response_model=SkillsUpdateRequest)
async def update_user_skills(skills_update: SkillsUpdateRequest, current_user: Annotated[dict, Depends(get_current_user)]):
    """Updates the user's skills list in their Firestore profile."""
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    try:
        # The 'update' method overwrites the specified field with the new data.
        db.collection("users").document(user_uid).update({"skills": skills_update.skills})
        print(f"Updated skills for UID: {user_uid}")
        return skills_update
    except Exception as e:
        print(f"Error updating skills for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not update skills")

@app.put("/api/profile/education", response_model=EducationUpdateRequest)
async def update_user_education(education_update: EducationUpdateRequest, current_user: Annotated[dict, Depends(get_current_user)]):
    """Updates the user's education list."""
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    try:
        # Convert the list of Pydantic models to a list of dictionaries suitable for Firestore.
        education_list_fs = [item.model_dump(by_alias=True) for item in education_update.education] 
        # Convert UUID objects to strings, as Firestore doesn't handle UUID type directly.
        for item_dict in education_list_fs: 
            item_dict['id'] = str(item_dict['id'])
        db.collection("users").document(user_uid).update({"education": education_list_fs})
        print(f"Updated education for UID: {user_uid}")
        return education_update
    except Exception as e:
        print(f"Error updating education for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not update education")

@app.put("/api/profile/experience", response_model=ExperienceUpdateRequest)
async def update_user_experience(experience_update: ExperienceUpdateRequest, current_user: Annotated[dict, Depends(get_current_user)]):
    """Updates the user's work experience list."""
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    try:
        experience_list_fs = [item.model_dump(by_alias=True) for item in experience_update.experience] 
        for item_dict in experience_list_fs: 
            item_dict['id'] = str(item_dict['id'])
        db.collection("users").document(user_uid).update({"experience": experience_list_fs})
        print(f"Updated experience for UID: {user_uid}")
        return experience_update
    except Exception as e:
        print(f"Error updating experience for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not update experience")


# --- The Main Job Matching Endpoint ---

@app.get("/api/jobs/match", response_model=List[MatchedJob])
async def get_job_matches(
    current_user: Annotated[dict, Depends(get_current_user)],
    min_score_threshold: float = 0.6, # A configurable threshold to filter out low-quality matches.
    # --- Weights for the Hybrid Score ---
    # These can be tuned to change the importance of different matching components.
    semantic_profile_weight: float = 0.3,
    keyword_weight: float = 0.3,        
    experience_level_weight: float = 0.25, 
    education_semantic_weight: float = 0.1, 
    education_presence_bonus: float = 0.02 # A small bonus for having any education listed.
):
    """
    Performs a hybrid search to find the best job matches for the current user.
    """
    user_uid = current_user.get("uid")
    print(f"\n=== Starting HYBRID Job Match (Detailed Insights) for User: {user_uid} ===")
    
    # Check that all required services and data are available before starting.
    if db is None or jobs_df is None or sentence_model is None or job_embeddings is None:
        raise HTTPException(status_code=503, detail="A required service or data is not available.")

    # --- Step 1: Fetch and Prepare User Profile Data ---
    try:
        user_doc = db.collection("users").document(user_uid).get()
        if not user_doc.exists: raise HTTPException(404, "User profile not found")
        
        user_profile_data = user_doc.to_dict()
        user_skills_list = user_profile_data.get("skills", [])
        user_experience_list_raw = user_profile_data.get("experience", [])
        user_education_list_raw = user_profile_data.get("education", [])
        
        # If the user's profile is completely empty, there's nothing to match on.
        if not any([user_skills_list, user_experience_list_raw, user_education_list_raw]):
            return []

        # Calculate the user's total years of experience and categorize it.
        user_total_years_exp = get_user_total_experience_years(user_experience_list_raw)
        user_experience_level_cat = categorize_user_experience(user_total_years_exp)
        
        # --- Create a semantic embedding for the user's main profile ---
        # Combine skills and past job titles into a single text block.
        user_profile_text_parts_main = []
        if user_skills_list: user_profile_text_parts_main.append("Key skills: " + ", ".join(user_skills_list) + ".")
        if user_experience_list_raw:
            exp_texts = [f"Professional experience as {exp.get('title', 'a role')}..." for exp in user_experience_list_raw if exp.get('title')]
            if exp_texts: user_profile_text_parts_main.append("Work history includes: " + " ".join(exp_texts))
        user_text_for_main_embedding = " ".join(user_profile_text_parts_main).strip() or "General profile"
        # Encode the user's profile text into a vector.
        user_main_embedding = sentence_model.encode(user_text_for_main_embedding, convert_to_tensor=True).unsqueeze(0)
        
        # --- Create separate embeddings for the user's education ---
        user_education_embeddings_tensor = None
        if user_education_list_raw:
            edu_texts = [f"{edu.get('degree','')}" for edu in user_education_list_raw if edu.get('degree')]
            if edu_texts:
                user_education_embeddings_tensor = sentence_model.encode(edu_texts, convert_to_tensor=True)
                # Ensure the tensor is 2D, even if there's only one education item.
                if user_education_embeddings_tensor.nelement() > 0 and len(user_education_embeddings_tensor.shape) == 1:
                    user_education_embeddings_tensor = user_education_embeddings_tensor.unsqueeze(0)
    except Exception as e:
        print(f"ERROR processing user profile: {e}")
        raise HTTPException(500, "Could not process user profile")

    # --- Step 2: Calculate Scores and Find Matches ---
    matched_jobs_output = []
    try:
        # **Performance Bottleneck**: This calculates similarity against ALL jobs.
        # This is where an ANN index would provide a significant speedup.
        all_main_semantic_scores_np = util.cos_sim(user_main_embedding, job_embeddings)[0].cpu().numpy()
        
        # Iterate through every job in the dataset.
        for idx, job_row in jobs_df.iterrows():
            # Calculate each component of the score using the helper functions.
            main_s = float(all_main_semantic_scores_np[idx])
            keyword_s = calculate_keyword_match_score(user_skills_list, job_row.get('requiredSkills', []))
            experience_s = calculate_experience_level_match_score(user_experience_level_cat, job_row.get('experience_level_required'))
            
            current_job_embedding = job_embeddings[idx]
            education_semantic_s = calculate_education_semantic_score(user_education_embeddings_tensor, current_job_embedding)
            
            edu_presence_b = education_presence_bonus if user_education_list_raw else 0.0
            
            # Combine the scores using the predefined weights.
            final_score = (semantic_profile_weight * main_s) + (keyword_weight * keyword_s) + \
                          (experience_level_weight * experience_s) + (education_semantic_weight * education_semantic_s) + edu_presence_b
            # Clamp the score to be between 0.0 and 1.0.
            final_score = min(max(final_score, 0.0), 1.0)
            
            # If the job's final score is above the threshold, add it to the results.
            if final_score >= min_score_threshold:
                # Find the specific keywords that matched for this job.
                user_skills_set_lower = set(s.lower() for s in user_skills_list)
                job_skills_set_lower = set(s.lower() for s in job_row.get('requiredSkills', []))
                matching_keywords_lower = user_skills_set_lower.intersection(job_skills_set_lower)
                matching_keywords_display = sorted([s for s in user_skills_list if s.lower() in matching_keywords_lower])
                
                try:
                    # Construct the final MatchedJob object.
                    job_data_dict = job_row.to_dict()
                    job_instance_data = {
                        "id": str(job_data_dict.get('id', '')), "title": job_data_dict.get('title', 'N/A'),
                        "company": job_data_dict.get('company'), "description": job_data_dict.get('description'),
                        "requiredSkills": job_data_dict.get('requiredSkills'), "experience_level_required": job_data_dict.get('experience_level_required'),
                        "matchScore": final_score,
                        "matching_keywords": matching_keywords_display,
                        "score_components": {
                            "semantic_profile_score": main_s, "keyword_skill_score": keyword_s,
                            "experience_level_score": experience_s, "education_semantic_score": education_semantic_s
                        }
                    }
                    matched_jobs_output.append(MatchedJob(**job_instance_data))
                except Exception as val_err:
                    print(f"[ERROR] Skipping job index {idx} due to Pydantic validation: {val_err}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR during job processing loop: {e}")
        raise HTTPException(status_code=500, detail="Error calculating job matches")
    
    # --- Step 3: Sort and Return Results ---
    # Sort the list of matched jobs by their score in descending order (best matches first).
    matched_jobs_output.sort(key=lambda job: job.matchScore, reverse=True)
    
    print(f"=== HYBRID Job Match (Detailed Insights) Finished. Found {len(matched_jobs_output)} matches. ===")
    return matched_jobs_output
