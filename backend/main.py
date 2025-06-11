# backend/main.py
import os
import pandas as pd
import ast 
from uuid import UUID, uuid4
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Annotated, Optional
from datetime import datetime, date 
from dotenv import load_dotenv
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
import re 

from sentence_transformers import SentenceTransformer, util
import torch

import firebase_admin
from firebase_admin import credentials, auth, firestore

# --- Load Environment Variables ---
load_dotenv()

# --- Firebase Admin SDK Initialization ---
db = None
try:
    SERVICE_ACCOUNT_PATH = "firebase-service-account.json"
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise FileNotFoundError(f"Service account key not found at {SERVICE_ACCOUNT_PATH}")
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    else:
        print("Firebase Admin SDK already initialized.")
    db = firestore.client()
    print("Firestore client obtained successfully.")
except Exception as e:
    print(f"CRITICAL ERROR initializing Firebase Admin SDK or Firestore client: {e}")

# --- Global Variables for ML Model and Data ---
jobs_df: Optional[pd.DataFrame] = None
JOBS_DATA_PATH = "data/jobs.csv" # Path to your IT dataset

MODEL_NAME = 'all-mpnet-base-v2' 
sentence_model: Optional[SentenceTransformer] = None
job_embeddings: Optional[torch.Tensor] = None

model_slug = MODEL_NAME.replace('/', '_').replace('-', '_')
EMBEDDINGS_CACHE_DIR = "data/cache"
# Updated cache name for this version of data processing
EMBEDDINGS_CACHE_PATH = os.path.join(EMBEDDINGS_CACHE_DIR, f"job_embeddings_{model_slug}_job_opportunities_dataset_v_refined_text.pt") 

# --- Helper Function to Load and Clean DataFrame ---
def load_and_clean_job_data(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        print(f"Attempting to load new job dataset from: {csv_path}...")
        if not os.path.exists(csv_path):
            print(f"ERROR: Job dataset file not found at {csv_path}.")
            return None
            
        temp_df = pd.read_csv(csv_path)
        print(f"Initial rows loaded from new CSV: {len(temp_df)}")

        temp_df.columns = [str(col).strip().lower().replace(' ', '_') for col in temp_df.columns]
        
        # Fill NaNs and ensure correct types
        temp_df['job_description'] = temp_df['job_description'].fillna('').astype(str)
        temp_df['role'] = temp_df['role'].fillna('N/A').astype(str) 
        temp_df['job_title'] = temp_df['job_title'].fillna('N/A').astype(str) 
        temp_df['company'] = temp_df['company'].fillna('N/A').astype(str)
        temp_df['experience'] = temp_df['experience'].fillna('Unknown').astype(str)
        # This 'skills' column from CSV contains sentences
        temp_df['skills_text_from_csv'] = temp_df.get('skills', pd.Series([''] * len(temp_df))).fillna('').astype(str) 

        def parse_experience_to_level(exp_str: str) -> str:
            # ... (same robust experience parsing logic as before) ...
            if pd.isna(exp_str) or exp_str.lower() in ['n/a', 'unknown', '']: return "Unknown"
            numbers = re.findall(r'\d+', exp_str)
            if not numbers:
                exp_str_lower = exp_str.lower()
                if "entry" in exp_str_lower or "fresher" in exp_str_lower: return "Entry-Level"
                if "junior" in exp_str_lower: return "Junior"
                if "senior" in exp_str_lower: return "Senior"
                if "lead" in exp_str_lower or "manager" in exp_str_lower: return "Senior"
                return "Unknown"
            try:
                years = int(numbers[0])
                if years < 1: return "Entry-Level"
                elif years < 3: return "Junior"
                elif years < 7: return "Mid-Level"
                else: return "Senior"
            except: return "Unknown"
        temp_df['parsed_experience_level'] = temp_df['experience'].apply(parse_experience_to_level)
        
        # Basic keyword extraction from the 'skills' (sentences) column for keyword matching
        # This list should ideally be much more comprehensive or replaced by NLP NER
        COMMON_TECH_KEYWORDS = ['java', 'python', 'sql', 'excel', 'aws', 'azure', 'cisco', 
                                'cybersecurity', 'pmp', 'agile', 'r', 'docker', 'jenkins', 
                                'react', 'angular', 'vue', 'node.js', 'javascript', 'typescript',
                                'html', 'css', 'tensorflow', 'pytorch', 'machine learning', 'data analysis'] # Example list
        def extract_keywords_from_sentences(text_data: str) -> List[str]:
            if pd.isna(text_data) or not isinstance(text_data, str): return []
            found_skills = set()
            text_data_lower = text_data.lower()
            for keyword in COMMON_TECH_KEYWORDS:
                # Use regex for whole word matching to avoid partial matches like 'r' in 'architecture'
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_data_lower):
                    found_skills.add(keyword.capitalize())
            return list(found_skills)
        temp_df['parsed_job_keywords'] = temp_df['skills_text_from_csv'].apply(extract_keywords_from_sentences)

        column_mappings = {
            'job_id': 'id', 'role': 'title', 'job_description': 'description',
            'company': 'company',
            'parsed_job_keywords': 'requiredSkills', # For keyword matching
            'parsed_experience_level': 'experience_level_required'
        }
        temp_df.rename(columns={k: v for k, v in column_mappings.items() if k in temp_df.columns}, inplace=True)
        
        if 'id' not in temp_df.columns:
            temp_df['id'] = [uuid4().hex for _ in range(len(temp_df))]
        else:
            temp_df['id'] = temp_df['id'].astype(str)
        
        # Ensure all Job model fields exist
        for field in ['title', 'company',  'description', 'requiredSkills', 'experience_level_required']:
            if field not in temp_df.columns:
                default_value = [] if field == 'requiredSkills' else \
                                'N/A' if field in ['company', 'title'] else \
                                '' if field == 'description' else \
                                'Unknown' if field == 'experience_level_required' else None
                temp_df[field] = default_value
        
        required_model_fields = ['id', 'title', 'company', 'description', 'requiredSkills', 'experience_level_required']
        final_columns = [col for col in required_model_fields if col in temp_df.columns]
        
        cleaned_df = temp_df[final_columns].copy()
        cleaned_df.dropna(subset=['id', 'title'], inplace=True) # Keep description even if empty for embedding

        if cleaned_df.empty:
            print("WARNING: DataFrame is empty after cleaning/filtering.")
            return None
        
        # --- Create text for JOB embedding (Role + Skills Sentences + Description) ---
        cleaned_df['title_for_emb'] = cleaned_df['title'].fillna('').astype(str)
        # Use original 'skills' column (sentences) from CSV for job embedding, standardized name is 'skills_text_from_csv'
        cleaned_df['skills_sentences_for_emb'] = temp_df.get('skills_text_from_csv', pd.Series([''] * len(cleaned_df))).fillna('').astype(str)
        cleaned_df['description_for_emb'] = cleaned_df['description'].fillna('').astype(str)
        
        cleaned_df['text_for_embedding'] = cleaned_df['title_for_emb'] + ". " + \
                                           cleaned_df['skills_sentences_for_emb'] + ". " + \
                                           cleaned_df['description_for_emb']
        # Clean up multiple periods or excessive whitespace if any
        cleaned_df['text_for_embedding'] = cleaned_df['text_for_embedding'].str.replace(r'\.\s*\.', '.', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
        print(f"Example text for JOB embedding: {cleaned_df['text_for_embedding'].iloc[0][:300] + '...' if not cleaned_df.empty and cleaned_df['text_for_embedding'].iloc[0] else 'N/A'}")
        
        print(f"Successfully cleaned and prepared {len(cleaned_df)} IT job postings.")
        return cleaned_df

    except Exception as e: # ... (error handling) ...
        print(f"ERROR: Failed to load or process IT job dataset from CSV: {e}")
        import traceback; traceback.print_exc()
        return None

# --- Load ML Model and Data (with Caching) ---
# ... (Same model loading and embedding caching logic as before, uses the refined load_and_clean_job_data) ...
try:
    print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    sentence_model = SentenceTransformer(MODEL_NAME)
    print(f"Sentence Transformer model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Sentence Transformer model '{MODEL_NAME}': {e}")
    sentence_model = None

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
            jobs_df = load_and_clean_job_data(JOBS_DATA_PATH)
            if jobs_df is None or len(jobs_df) != job_embeddings.shape[0]:
                print(f"WARNING: Mismatch or error loading DataFrame with cache. Recomputing embeddings.")
                job_embeddings = None; jobs_df = None 
        except Exception as e:
            print(f"ERROR loading embeddings from cache or verifying DataFrame: {e}. Recomputing.")
            job_embeddings = None; jobs_df = None
            
    if job_embeddings is None: 
        print(f"Recomputing job embeddings for {MODEL_NAME}...")
        if jobs_df is None: jobs_df = load_and_clean_job_data(JOBS_DATA_PATH)
        
        if jobs_df is not None and not jobs_df.empty:
            if 'text_for_embedding' not in jobs_df.columns or jobs_df['text_for_embedding'].isnull().all():
                print("ERROR: 'text_for_embedding' column is missing or all null in jobs_df. Cannot compute embeddings.")
                job_embeddings = None
            else:
                job_texts_for_embedding = jobs_df['text_for_embedding'].fillna('').tolist()
                try:
                    job_embeddings = sentence_model.encode(job_texts_for_embedding, convert_to_tensor=True, show_progress_bar=True)
                    print(f"Job embeddings calculated. Shape: {job_embeddings.shape}. Saving to {EMBEDDINGS_CACHE_PATH}...")
                    torch.save(job_embeddings, EMBEDDINGS_CACHE_PATH)
                    print("Job embeddings saved to cache.")
                except Exception as e:
                    print(f"ERROR calculating or saving embeddings: {e}"); job_embeddings = None
        else:
            print("Jobs DataFrame empty/not loaded; cannot compute embeddings."); job_embeddings = None
else:
    print("WARNING: Sentence Transformer model not loaded. Semantic matching will be unavailable.")
    if jobs_df is None: jobs_df = load_and_clean_job_data(JOBS_DATA_PATH)

# --- Pydantic Models ---
# ... (SkillsUpdate, EducationItem, EducationUpdate, ExperienceItem, ExperienceUpdate - same as before) ...
class SkillsUpdateRequest(BaseModel): skills: List[str] = Field(..., example=["Python"])
class EducationItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    degree: str; school: str
    startYear: Optional[int] = None; endYear: Optional[int] = None
class EducationUpdateRequest(BaseModel): education: List[EducationItem]
class ExperienceItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str; company: str
    startDate: Optional[str] = None; endDate: Optional[str] = None 
class ExperienceUpdateRequest(BaseModel): experience: List[ExperienceItem]
class ScoreComponents(BaseModel):
    semantic_profile_score: float
    keyword_skill_score: float
    experience_level_score: float
    education_semantic_score: float

class Job(BaseModel):
    id: str; title: str
    company: Optional[str] = None
    description: Optional[str] = ""
    requiredSkills: List[str] = Field(default_factory=list)
    experience_level_required: Optional[str] = None 
class MatchedJob(Job): # Modified
    matchScore: float = Field(..., example=0.75)
    score_components: ScoreComponents
    matching_keywords: List[str] = Field(default_factory=list)
class UserProfileResponse(BaseModel):
    uid: str; email: Optional[str] = None; displayName: Optional[str] = None
    photoURL: Optional[str] = None; createdAt: Optional[datetime] = None
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    saved_job_ids: List[str] = Field(default_factory=list) # Added for Saved Jobs feature

# --- FastAPI app instance & CORS ---
# ... (same) ...
app = FastAPI()
# ... (All helper functions and endpoints from your code, with modifications below) ...
origins = ["http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def get_current_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    if authorization is None: raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Auth header missing")
    parts = authorization.split();
    if len(parts) != 2 or parts[0].lower() != "bearer": raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid auth header")
    id_token = parts[1]
    try: return auth.verify_id_token(id_token, check_revoked=True)
    except Exception as e: print(f"Token verification failed: {e}"); raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not verify token")

def calculate_keyword_match_score(user_skills: List[str], job_skills: List[str]) -> float:
    if not job_skills: return 0.0
    user_s = set(s.lower() for s in user_skills); job_s = set(s.lower() for s in job_skills)
    return len(user_s.intersection(job_s)) / len(job_s) if len(job_s) > 0 else 0.0

def get_user_total_experience_years(experience_list: List[dict]) -> float:
    total_years = 0.0; today = date.today()
    for exp_item_dict in experience_list:
        try:
            start_date_str = exp_item_dict.get('startDate'); end_date_str = exp_item_dict.get('endDate')
            if not start_date_str: continue
            start_date_obj = parse_date(start_date_str, default=datetime(int(start_date_str[:4]), int(start_date_str[5:7]), 1) if len(start_date_str) == 7 else None).date()
            end_date_obj = today
            if end_date_str and end_date_str.lower().strip() != 'present' and end_date_str.strip() != '':
                try: end_date_obj = parse_date(end_date_str, default=datetime(int(end_date_str[:4]), int(end_date_str[5:7]), 1) if len(end_date_str) == 7 else None).date()
                except (ValueError, TypeError): print(f"Could not parse end date: '{end_date_str}', assuming 'Present'.")
            if end_date_obj < start_date_obj: print(f"Warning: End date {end_date_obj} < start {start_date_obj} for exp: {exp_item_dict.get('title')}"); continue
            delta = relativedelta(end_date_obj, start_date_obj)
            total_years += delta.years + (delta.months / 12.0) + (delta.days / 365.25)
        except Exception as e: print(f"Warning: Could not parse dates for exp item '{exp_item_dict.get('title')}': {e}")
    return total_years

def categorize_user_experience(total_years: float) -> str:
    if total_years < 1: return "Entry-Level"; 
    elif total_years < 3: return "Junior";
    elif total_years < 7: return "Mid-Level"; 
    else: return "Senior"

def calculate_experience_level_match_score(user_level_str: str, job_level_str: Optional[str]) -> float:
    if not job_level_str or pd.isna(job_level_str) or job_level_str.lower().strip() in ['n/a', 'unknown', '']: return 0.5 
    user_level_norm = user_level_str.lower().replace('-', '').replace(' ', ''); job_level_norm = job_level_str.lower().replace('-', '').replace(' ', '')
    level_map = {"entrylevel": 0, "junior": 1, "midlevel": 2, "senior": 3}
    user_val = level_map.get(user_level_norm, -1); job_val = level_map.get(job_level_norm, -1)
    if user_val == -1 or job_val == -1: return 0.25 
    if user_val == job_val: return 1.0
    elif user_val > job_val:
        diff = user_val - job_val
        if diff == 1: return 0.7; 
        elif diff == 2: return 0.4; 
        else: return 0.2
    elif job_val == user_val + 1: return 0.6 
    elif job_val == user_val + 2: return 0.2
    else: return 0.05

def calculate_education_semantic_score(user_education_embeddings: Optional[torch.Tensor], job_text_embedding: torch.Tensor) -> float:
    if user_education_embeddings is None or user_education_embeddings.nelement() == 0 or job_text_embedding is None: return 0.0
    max_edu_score = 0.0
    try:
        current_job_embedding_expanded = job_text_embedding
        if len(job_text_embedding.shape) == 1: current_job_embedding_expanded = job_text_embedding.unsqueeze(0)
        if not (torch.isnan(user_education_embeddings).any() or torch.isnan(current_job_embedding_expanded).any()):
            cosine_scores_tensor = util.cos_sim(user_education_embeddings, current_job_embedding_expanded)
            if cosine_scores_tensor.numel() > 0: max_edu_score = float(torch.max(cosine_scores_tensor).item())
    except Exception as e: print(f"[WARN] Error calculating education semantic score: {e}")
    return max(0.0, max_edu_score)
# ...
# --- Profile Endpoints ---
# ... (GET /, GET /api/profile (with modification), PUT skills, edu, exp - same as your code) ...
@app.get("/api/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None:
        raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    user_email = current_user.get("email", "")
    user_display_name = current_user.get("name", "")
    user_photo_url = current_user.get("picture", "")
    user_created_at = datetime.fromtimestamp(current_user.get("auth_time", 0))

    try:
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()

        if not user_doc.exists:
            # This creates the Firestore document on first access
            user_doc_ref.set({
                "email": user_email,
                "displayName": user_display_name,
                "photoURL": user_photo_url,
                "createdAt": user_created_at,
                "skills": [],
                "education": [],
                "experience": [],
                "saved_job_ids": []
            })
            print(f"Created Firestore profile for UID: {user_uid}")

        profile_data = user_doc_ref.get().to_dict()
        profile_data["uid"] = user_uid
        profile_data.setdefault("saved_job_ids", [])

        return UserProfileResponse(**profile_data)
    except Exception as e:
        print(f"Error fetching or creating profile for {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch or create user profile")

# --- NEW ENDPOINTS FOR SAVED JOBS ---
@app.post("/api/profile/saved-jobs/{job_id}", status_code=status.HTTP_200_OK)
async def save_job(job_id: str, current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    user_doc_ref = db.collection("users").document(user_uid)
    try:
        # array_union appends the job_id only if it's not already present
        user_doc_ref.update({"saved_job_ids": firestore.ArrayUnion([job_id])})
        return {"status": "success", "message": f"Job {job_id} saved."}
    except Exception as e:
        print(f"Error saving job {job_id} for user {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not save job.")

@app.delete("/api/profile/saved-jobs/{job_id}", status_code=status.HTTP_200_OK)
async def unsave_job(job_id: str, current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    user_doc_ref = db.collection("users").document(user_uid)
    try:
        # array_remove deletes all instances of the job_id from the array
        user_doc_ref.update({"saved_job_ids": firestore.ArrayRemove([job_id])})
        return {"status": "success", "message": f"Job {job_id} unsaved."}
    except Exception as e:
        print(f"Error unsaving job {job_id} for user {user_uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not unsave job.")

@app.get("/api/profile/saved-jobs", response_model=List[Job])
async def get_saved_jobs(current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    if jobs_df is None: raise HTTPException(status_code=503, detail="Job data is not available")
    user_uid = current_user.get("uid")
    try:
        user_doc = db.collection("users").document(user_uid).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        user_data = user_doc.to_dict()
        saved_ids = user_data.get("saved_job_ids", [])
        if not saved_ids:
            return [] 

        saved_jobs_df = jobs_df[jobs_df['id'].isin(saved_ids)]
        saved_jobs_list = saved_jobs_df.to_dict(orient='records')
        
        saved_jobs_dict = {job['id']: job for job in saved_jobs_list}
        ordered_saved_jobs = [saved_jobs_dict[job_id] for job_id in saved_ids if job_id in saved_jobs_dict]

        return ordered_saved_jobs
    except Exception as e:
        print(f"Error fetching saved jobs for user {user_uid}: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve saved jobs.")

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
        education_list_fs = [item.model_dump(by_alias=True) for item in education_update.education] 
        for item_dict in education_list_fs: 
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
        experience_list_fs = [item.model_dump(by_alias=True) for item in experience_update.experience] 
        for item_dict in experience_list_fs: 
            item_dict['id'] = str(item_dict['id'])
        db.collection("users").document(user_uid).update({"experience": experience_list_fs})
        print(f"Updated experience for UID: {user_uid}")
        return experience_update
    except Exception as e:
        print(f"Error updating experience for {user_uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not update experience")


# --- Job Matching Endpoint (Updated for refined text embedding) ---
@app.get("/api/jobs/match", response_model=List[MatchedJob])
async def get_job_matches(
    current_user: Annotated[dict, Depends(get_current_user)],
    min_score_threshold: float = 0.6, # Example tuned threshold
    semantic_profile_weight: float = 0.3,
    keyword_weight: float = 0.3,        
    experience_level_weight: float = 0.25, 
    education_semantic_weight: float = 0.1, 
    education_presence_bonus: float = 0.02
):
    user_uid = current_user.get("uid")
    print(f"\n=== Starting HYBRID Job Match (Detailed Insights) for User: {user_uid} ===")
    
    if db is None or jobs_df is None or sentence_model is None or job_embeddings is None:
        raise HTTPException(status_code=503, detail="A required service or data is not available.")

    # 1. Fetch & Prepare User Profile
    try:
        user_doc = db.collection("users").document(user_uid).get()
        if not user_doc.exists: raise HTTPException(404, "User profile not found")
        user_profile_data = user_doc.to_dict()
        user_skills_list = user_profile_data.get("skills", [])
        user_experience_list_raw = user_profile_data.get("experience", [])
        user_education_list_raw = user_profile_data.get("education", [])
        
        if not any([user_skills_list, user_experience_list_raw, user_education_list_raw]):
            return [] # Return early for empty profiles

        user_total_years_exp = get_user_total_experience_years(user_experience_list_raw)
        user_experience_level_cat = categorize_user_experience(user_total_years_exp)
        
        user_profile_text_parts_main = []
        if user_skills_list: user_profile_text_parts_main.append("Key skills: " + ", ".join(user_skills_list) + ".")
        if user_experience_list_raw:
            exp_texts = [f"Professional experience as {exp.get('title', 'a role')}..." for exp in user_experience_list_raw if exp.get('title')]
            if exp_texts: user_profile_text_parts_main.append("Work history includes: " + " ".join(exp_texts))
        user_text_for_main_embedding = " ".join(user_profile_text_parts_main).strip() or "General profile"
        user_main_embedding = sentence_model.encode(user_text_for_main_embedding, convert_to_tensor=True).unsqueeze(0)
        
        user_education_embeddings_tensor = None
        if user_education_list_raw:
            edu_texts = [f"{edu.get('degree','')}" for edu in user_education_list_raw if edu.get('degree')]
            if edu_texts:
                user_education_embeddings_tensor = sentence_model.encode(edu_texts, convert_to_tensor=True)
                if user_education_embeddings_tensor.nelement() > 0 and len(user_education_embeddings_tensor.shape) == 1:
                    user_education_embeddings_tensor = user_education_embeddings_tensor.unsqueeze(0)
    except Exception as e: print(f"ERROR processing user profile: {e}"); raise HTTPException(500, "Could not process user profile")

    # 2. Calculate Scores and Match
    matched_jobs_output = []
    try:
        all_main_semantic_scores_np = util.cos_sim(user_main_embedding, job_embeddings)[0].cpu().numpy()
        
        for idx, job_row in jobs_df.iterrows():
            main_s = float(all_main_semantic_scores_np[idx])
            keyword_s = calculate_keyword_match_score(user_skills_list, job_row.get('requiredSkills', []))
            experience_s = calculate_experience_level_match_score(user_experience_level_cat, job_row.get('experience_level_required'))
            
            # Using the main job embedding for education semantic score as a fallback if specific skills embedding isn't available
            current_job_embedding = job_embeddings[idx]
            education_semantic_s = calculate_education_semantic_score(user_education_embeddings_tensor, current_job_embedding)
            
            edu_presence_b = education_presence_bonus if user_education_list_raw else 0.0
            
            final_score = (semantic_profile_weight * main_s) + (keyword_weight * keyword_s) + \
                          (experience_level_weight * experience_s) + (education_semantic_weight * education_semantic_s) + edu_presence_b
            final_score = min(max(final_score, 0.0), 1.0)
            
            if final_score >= min_score_threshold:
                user_skills_set_lower = set(s.lower() for s in user_skills_list)
                job_skills_set_lower = set(s.lower() for s in job_row.get('requiredSkills', []))
                matching_keywords_lower = user_skills_set_lower.intersection(job_skills_set_lower)
                matching_keywords_display = sorted([s for s in user_skills_list if s.lower() in matching_keywords_lower])
                
                try:
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
                except Exception as val_err: print(f"[ERROR] Skipping job index {idx} due to Pydantic validation: {val_err}")

    except Exception as e: print(f"ERROR during job processing loop: {e}"); raise HTTPException(status_code=500, detail="Error calculating job matches")
    
    matched_jobs_output.sort(key=lambda job: job.matchScore, reverse=True)
    print(f"=== HYBRID Job Match (Detailed Insights) Finished. Found {len(matched_jobs_output)} matches. ===")
    return matched_jobs_output
