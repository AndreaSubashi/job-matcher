# backend/main.py
import os
import pandas as pd
import ast # For parsing stringified lists like '["skill1", "skill2"]'
from uuid import UUID, uuid4
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Annotated, Optional
from datetime import datetime
from dotenv import load_dotenv # For environment variables

from sentence_transformers import SentenceTransformer, util # For sentence embeddings
import torch # For PyTorch tensors (embeddings)

import firebase_admin
from firebase_admin import credentials, auth, firestore

# --- Load Environment Variables ---
load_dotenv() # Load from .env file at the root of the 'backend' folder

# --- Firebase Admin SDK Initialization ---
db = None # Initialize db to None
try:
    # Ensure firebase-service-account.json is in the backend directory
    # or provide the correct path.
    SERVICE_ACCOUNT_PATH = "firebase-service-account.json"
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise FileNotFoundError(f"Service account key not found at {SERVICE_ACCOUNT_PATH}")

    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    if not firebase_admin._apps: # Check if already initialized
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    else:
        print("Firebase Admin SDK already initialized.")
    db = firestore.client()
    print("Firestore client obtained successfully.")
except Exception as e:
    print(f"CRITICAL ERROR initializing Firebase Admin SDK or Firestore client: {e}")
    # Depending on the error, you might want the app to not start or run in a degraded mode.
    # For now, db will remain None if this fails.

# --- Global Variables for ML Model and Data ---
jobs_df: Optional[pd.DataFrame] = None
JOBS_DATA_PATH = "data/job_skill_set.csv" # Adjust if your CSV is elsewhere or named differently

#MODEL_NAME = 'all-MiniLM-L6-v2' # Faster model, good for development
MODEL_NAME = 'all-mpnet-base-v2' # Larger, potentially more accurate, slower startup without cache

sentence_model: Optional[SentenceTransformer] = None
job_embeddings: Optional[torch.Tensor] = None

# --- Define Cache Path based on Model ---
model_slug = MODEL_NAME.replace('/', '_').replace('-', '_') # Create a file-safe slug from model name
EMBEDDINGS_CACHE_DIR = "data/cache"
EMBEDDINGS_CACHE_PATH = os.path.join(EMBEDDINGS_CACHE_DIR, f"job_embeddings_{model_slug}.pt")

# --- Helper Function to Load and Clean DataFrame ---
def load_and_clean_job_data(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        print(f"Attempting to load job dataset from: {csv_path}...")
        if not os.path.exists(csv_path):
            print(f"ERROR: Job dataset file not found at {csv_path}.")
            return None
            
        temp_df = pd.read_csv(csv_path)
        print(f"Initial rows loaded from CSV: {len(temp_df)}")

        # Data Cleaning / Preparation
        temp_df['job_description'] = temp_df['job_description'].fillna('')
        temp_df['job_title'] = temp_df['job_title'].fillna('N/A')

        SKILLS_COLUMN_NAME = 'job_skill_set' # <-- ADJUST IF YOUR CSV SKILLS COLUMN IS DIFFERENT
        if SKILLS_COLUMN_NAME not in temp_df.columns:
            print(f"WARNING: Skills column '{SKILLS_COLUMN_NAME}' not found. Adding empty 'requiredSkills' list.")
            temp_df['requiredSkills'] = pd.Series([[] for _ in range(len(temp_df))])
        else:
            print(f"Parsing skills from column: '{SKILLS_COLUMN_NAME}'...")
            def parse_skill_list(skill_str):
                if pd.isna(skill_str) or not isinstance(skill_str, str) or not skill_str.strip().startswith('['):
                    # Handle cases that are not string representations of lists or are NaN
                    # Also, if it's a comma-separated string NOT in list format, handle that here
                    # e.g. if skill_str == "Python, Java": return [s.strip() for s in skill_str.split(',') if s.strip()]
                    return []
                try:
                    skills = ast.literal_eval(skill_str) # For strings like '["Python", "Java"]'
                    return [str(s).strip() for s in skills if str(s).strip()]
                except (ValueError, SyntaxError, TypeError) as parse_error:
                    # print(f"Warning: Could not parse skill string: '{skill_str}' | Error: {parse_error}")
                    return []
            temp_df['requiredSkills'] = temp_df[SKILLS_COLUMN_NAME].apply(parse_skill_list)

        column_mappings = {
            'job_id': 'id', 
            'job_title': 'title',
            'job_description': 'description',
            # Example: If your CSV has 'Company Name' and 'Work Location'
            # 'Company Name': 'company',
            # 'Work Location': 'location',
        }
        actual_mappings = {k: v for k, v in column_mappings.items() if k in temp_df.columns}
        temp_df.rename(columns=actual_mappings, inplace=True)

        if 'id' not in temp_df.columns:
            print("WARNING: 'id' column not found after mapping, generating new UUIDs.")
            temp_df['id'] = [uuid4().hex for _ in range(len(temp_df))]
        else:
            temp_df['id'] = temp_df['id'].astype(str)

        # Ensure Job model fields exist, add defaults
        if 'company' not in temp_df.columns: temp_df['company'] = 'N/A'
        if 'location' not in temp_df.columns: temp_df['location'] = None
        if 'description' not in temp_df.columns: temp_df['description'] = ''
        if 'requiredSkills' not in temp_df.columns: # Should be created by parsing logic
            temp_df['requiredSkills'] = pd.Series([[] for _ in range(len(temp_df))])


        required_model_fields = ['id', 'title', 'company', 'location', 'description', 'requiredSkills']
        final_columns = [col for col in required_model_fields if col in temp_df.columns]
        
        cleaned_df = temp_df[final_columns].copy()
        cleaned_df.dropna(subset=['id', 'title', 'description'], inplace=True)

        if cleaned_df.empty:
            print("WARNING: DataFrame is empty after cleaning/filtering.")
            return None
        
        print(f"Successfully cleaned and prepared {len(cleaned_df)} job postings.")
        return cleaned_df

    except Exception as e:
        print(f"ERROR: Failed to load or process job dataset from CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Load ML Model on Startup ---
try:
    print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    sentence_model = SentenceTransformer(MODEL_NAME)
    print(f"Sentence Transformer model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Sentence Transformer model '{MODEL_NAME}': {e}")
    sentence_model = None

# --- Load Job DataFrame and Embeddings (with Caching) ---
if sentence_model:
    try:
        os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create cache directory '{EMBEDDINGS_CACHE_DIR}': {e}")

    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        print(f"Attempting to load cached job embeddings from {EMBEDDINGS_CACHE_PATH}...")
        try:
            job_embeddings = torch.load(EMBEDDINGS_CACHE_PATH)
            print(f"Successfully loaded job embeddings from cache. Shape: {job_embeddings.shape}")
            jobs_df = load_and_clean_job_data(JOBS_DATA_PATH) # Load DF for metadata
            
            if jobs_df is None or len(jobs_df) != job_embeddings.shape[0]:
                print(f"WARNING: Mismatch or error loading DataFrame with cache. DataFrame rows: {len(jobs_df) if jobs_df is not None else 'None'}, Cached embeddings: {job_embeddings.shape[0]}. Recomputing.")
                job_embeddings = None # Force recompute
                jobs_df = None # Reset jobs_df to reload fully for recomputation
        except Exception as e:
            print(f"ERROR loading embeddings from cache or verifying DataFrame: {e}. Recomputing.")
            job_embeddings = None
            jobs_df = None
            
    if job_embeddings is None: # If cache didn't exist, mismatch, or failed load
        print(f"No valid cache found or issue with cached data. Recomputing job embeddings for {MODEL_NAME}...")
        if jobs_df is None: # Ensure jobs_df is loaded if previous step failed
            jobs_df = load_and_clean_job_data(JOBS_DATA_PATH)

        if jobs_df is not None and not jobs_df.empty:
            print(f"Preparing to calculate embeddings for {len(jobs_df)} job descriptions...")
            job_texts_for_embedding = jobs_df['description'].tolist()
            try:
                job_embeddings = sentence_model.encode(job_texts_for_embedding, convert_to_tensor=True, show_progress_bar=True)
                print(f"Job embeddings calculated. Shape: {job_embeddings.shape}. Saving to {EMBEDDINGS_CACHE_PATH}...")
                torch.save(job_embeddings, EMBEDDINGS_CACHE_PATH)
                print("Job embeddings saved to cache.")
            except Exception as e:
                print(f"ERROR calculating or saving embeddings: {e}")
                job_embeddings = None
        else:
            print("Jobs DataFrame is empty or not loaded; cannot compute embeddings.")
            job_embeddings = None
else:
    print("WARNING: Sentence Transformer model not loaded. Semantic matching will be unavailable.")
    if jobs_df is None: # Attempt to load jobs_df for non-semantic features if model failed
        jobs_df = load_and_clean_job_data(JOBS_DATA_PATH)

# --- Pydantic Models ---
class SkillsUpdateRequest(BaseModel):
    skills: List[str] = Field(..., example=["Python", "React", "SQL"])

class EducationItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    degree: str = Field(..., example="Bachelor of Science")
    school: str = Field(..., example="University of Example")
    startYear: Optional[int] = Field(None, example=2018)
    endYear: Optional[int] = Field(None, example=2022)

class EducationUpdateRequest(BaseModel):
    education: List[EducationItem]

class ExperienceItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., example="Software Engineer")
    company: str = Field(..., example="Tech Solutions Inc.")
    startDate: Optional[str] = Field(None, example="2020-01")
    endDate: Optional[str] = Field(None, example="2022-12")
    location: Optional[str] = Field(None, example="Remote")
    description: Optional[str] = Field(None, example="Developed features for...")

class ExperienceUpdateRequest(BaseModel):
    experience: List[ExperienceItem]

class Job(BaseModel):
    id: str
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = ""
    requiredSkills: List[str] = Field(default_factory=list)

class MatchedJob(Job):
    matchScore: float = Field(..., example=0.75)

class UserProfileResponse(BaseModel):
    uid: str
    email: Optional[str] = None
    displayName: Optional[str] = None
    photoURL: Optional[str] = None
    createdAt: Optional[datetime] = None
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)

# --- FastAPI app instance ---
app = FastAPI()

# --- CORS Middleware ---
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication Dependency ---
async def get_current_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    if authorization is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format. Use 'Bearer <token>'")
    id_token = parts[1]
    try:
        decoded_token = auth.verify_id_token(id_token, check_revoked=True)
        return decoded_token
    except auth.ExpiredIdTokenError: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="ID token has expired",headers={"WWW-Authenticate": "Bearer"})
    except auth.RevokedIdTokenError: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="ID token has been revoked",headers={"WWW-Authenticate": "Bearer"})
    except auth.InvalidIdTokenError: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Invalid ID token",headers={"WWW-Authenticate": "Bearer"})
    except Exception as e:
        print(f"Token verification failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="Could not verify token")

# --- Keyword Match Score Helper ---
def calculate_match_score(user_skills: List[str], job_skills: List[str]) -> float:
    if not job_skills: return 0.0
    user_skills_set = set(s.lower() for s in user_skills)
    job_skills_set = set(s.lower() for s in job_skills)
    matching_skills_count = len(user_skills_set.intersection(job_skills_set))
    score = matching_skills_count / len(job_skills_set)
    # print(f"[Keyword Score] User: {user_skills_set}, Job: {job_skills_set}, Match: {matching_skills_count}, Score: {score:.2f}")
    return score

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Resume Analyzer Backend is running!"}

@app.get("/api/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    try:
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()
        if user_doc.exists:
            profile_data = user_doc.to_dict()
            profile_data["uid"] = user_uid # Ensure UID is part of the response data
            return UserProfileResponse(**profile_data)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found")
    except Exception as e:
        print(f"Error fetching profile for {user_uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not fetch profile")

@app.put("/api/profile/skills", response_model=SkillsUpdateRequest)
async def update_user_skills(skills_update: SkillsUpdateRequest, current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    try:
        db.collection("users").document(user_uid).update({"skills": skills_update.skills})
        print(f"Updated skills for UID: {user_uid}")
        return skills_update
    except Exception as e:
        print(f"Error updating skills for {user_uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not update skills")

@app.put("/api/profile/education", response_model=EducationUpdateRequest)
async def update_user_education(education_update: EducationUpdateRequest, current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    try:
        education_list_fs = [item.model_dump(by_alias=True) for item in education_update.education] # Use model_dump
        for item_dict in education_list_fs: # Ensure ID is string
            item_dict['id'] = str(item_dict['id'])
        db.collection("users").document(user_uid).update({"education": education_list_fs})
        print(f"Updated education for UID: {user_uid}")
        return education_update
    except Exception as e:
        print(f"Error updating education for {user_uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not update education")

@app.put("/api/profile/experience", response_model=ExperienceUpdateRequest)
async def update_user_experience(experience_update: ExperienceUpdateRequest, current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    try:
        experience_list_fs = [item.model_dump(by_alias=True) for item in experience_update.experience] # Use model_dump
        for item_dict in experience_list_fs: # Ensure ID is string
            item_dict['id'] = str(item_dict['id'])
        db.collection("users").document(user_uid).update({"experience": experience_list_fs})
        print(f"Updated experience for UID: {user_uid}")
        return experience_update
    except Exception as e:
        print(f"Error updating experience for {user_uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not update experience")

# --- Job Matching Endpoint (Semantic) ---
@app.get("/api/jobs/match", response_model=List[MatchedJob])
async def get_job_matches(
    current_user: Annotated[dict, Depends(get_current_user)],
    min_score_threshold: float = 0.3, # Threshold for semantic scores
    # Add weights here if/when hybrid scoring is implemented
    # semantic_weight: float = 0.7,
    # keyword_weight: float = 0.3
):
    user_uid = current_user.get("uid")
    print(f"\n=== Starting SEMANTIC Job Match for User: {user_uid} ===")

    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable (db)")
    if jobs_df is None or jobs_df.empty:
        print("[ERROR] Job dataset (DataFrame) not loaded or empty. Cannot perform matching.")
        return []
    if sentence_model is None:
        print("[ERROR] Sentence Transformer model not loaded. Cannot perform semantic match.")
        raise HTTPException(status_code=503, detail="Matching service model unavailable")
    if job_embeddings is None or len(job_embeddings) != len(jobs_df):
         print("[ERROR] Job embeddings not pre-computed or mismatch length. Cannot perform semantic match.")
         raise HTTPException(status_code=503, detail="Job embedding data unavailable")

    try:
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()
        if not user_doc.exists: raise HTTPException(404, "User profile not found")
        user_profile_data = user_doc.to_dict()
        user_skills_list = user_profile_data.get("skills", [])
        # Consider adding user experience text here for richer embedding
        user_experience_list = user_profile_data.get("experience", [])
        
        user_profile_text_parts = []
        if user_skills_list:
            user_profile_text_parts.append("Skills: " + ", ".join(user_skills_list) + ".")
        if user_experience_list:
            exp_texts = []
            for exp in user_experience_list:
                 exp_text = f"Worked as {exp.get('title', '')} at {exp.get('company', '')}."
                 if exp.get('description'): exp_text += f" Responsibilities: {exp.get('description')}."
                 exp_texts.append(exp_text)
            if exp_texts: user_profile_text_parts.append("Experience: " + " ".join(exp_texts))
        
        user_text_for_embedding = " ".join(user_profile_text_parts).strip()

        if not user_text_for_embedding:
             print("[INFO] User profile text for embedding is empty (no skills/experience). Returning no matches.")
             return []
        print(f"[DEBUG] User text for embedding: '{user_text_for_embedding}'")

    except Exception as e:
        print(f"ERROR fetching user profile data for embedding for {user_uid}: {e}")
        raise HTTPException(500, "Could not process user profile for matching")

    matched_jobs_output = []
    try:
        print("[DEBUG] Calculating user profile embedding...")
        user_embedding = sentence_model.encode(user_text_for_embedding, convert_to_tensor=True)
        
        # If user_embedding is 1D, unsqueeze for cos_sim with batch of job_embeddings
        if len(user_embedding.shape) == 1:
            user_embedding = user_embedding.unsqueeze(0)

        print("[DEBUG] Calculating cosine similarities...")
        # job_embeddings should be a 2D tensor [num_jobs, embedding_dim]
        cosine_scores_tensor = util.cos_sim(user_embedding, job_embeddings)[0] # Get the first row of scores
        cosine_scores_np = cosine_scores_tensor.cpu().numpy()

        print(f"[DEBUG] Calculated {len(cosine_scores_np)} semantic scores.")
        
        for idx, semantic_score_float in enumerate(cosine_scores_np):
            final_score = float(semantic_score_float) # Start with semantic score
            
            # --- Placeholder for Hybrid Score Calculation ---
            # job_required_skills = jobs_df.iloc[idx].get('requiredSkills', [])
            # keyword_s = calculate_match_score(user_skills_list, job_required_skills)
            # final_score = (semantic_weight * semantic_score_float) + (keyword_weight * keyword_s)
            # print(f"[DEBUG] Job ID {jobs_df.iloc[idx].get('id', 'N/A')}: SemScore={semantic_score_float:.2f}, KeyScore={keyword_s:.2f}, FinalScore={final_score:.2f}")
            # --- End Placeholder ---
            
            if final_score >= min_score_threshold:
                try:
                    job_row = jobs_df.iloc[idx]
                    job_data_dict = job_row.to_dict()
                    
                    job_instance_data = {
                        "id": str(job_data_dict.get('id', '')),
                        "title": job_data_dict.get('title', 'N/A'),
                        "company": job_data_dict.get('company', None),
                        "location": job_data_dict.get('location', None),
                        "description": job_data_dict.get('description', ""),
                        "requiredSkills": job_data_dict.get('requiredSkills', []) if isinstance(job_data_dict.get('requiredSkills'), list) else [],
                        "matchScore": final_score
                    }
                    matched_jobs_output.append(MatchedJob(**job_instance_data))
                except Exception as val_err:
                    print(f"[ERROR] Skipping job index {idx} due to Pydantic validation: {val_err}")

    except Exception as e:
        print(f"ERROR during semantic similarity or job processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error calculating job matches")

    matched_jobs_output.sort(key=lambda job: job.matchScore, reverse=True)
    print(f"=== SEMANTIC Job Match Finished. Found {len(matched_jobs_output)} matches. ===")
    return matched_jobs_output