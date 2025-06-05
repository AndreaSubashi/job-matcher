# backend/main.py
import os
import pandas as pd
import ast 
from uuid import UUID, uuid4 # Ensure uuid4 is imported
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Annotated, Optional
from datetime import datetime, date 
from dotenv import load_dotenv
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
import re # For parsing experience strings

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
JOBS_DATA_PATH = "data/jobs.csv" # PATH TO YOUR NEW CSV

MODEL_NAME = 'all-mpnet-base-v2' 
sentence_model: Optional[SentenceTransformer] = None
job_embeddings: Optional[torch.Tensor] = None

model_slug = MODEL_NAME.replace('/', '_').replace('-', '_')
EMBEDDINGS_CACHE_DIR = "data/cache"
EMBEDDINGS_CACHE_PATH = os.path.join(EMBEDDINGS_CACHE_DIR, f"job_embeddings_{model_slug}_job_opportunities_dataset_v2.pt") # Incremented cache name

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
        
        temp_df['job_description'] = temp_df['job_description'].fillna('').astype(str)
        temp_df['role'] = temp_df['role'].fillna('N/A').astype(str) 
        temp_df['job_title'] = temp_df['job_title'].fillna('N/A').astype(str) 
        temp_df['company'] = temp_df['company'].fillna('N/A').astype(str)
        temp_df['experience'] = temp_df['experience'].fillna('Unknown').astype(str)
        temp_df['skills'] = temp_df['skills'].fillna('').astype(str)

        def parse_experience_to_level(exp_str: str) -> str:
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
        
        CSV_SKILLS_COLUMN_STANDARDIZED = 'skills' 
        print(f"Attempting basic skill extraction from sentence-based column: '{CSV_SKILLS_COLUMN_STANDARDIZED}'...")
        COMMON_TECH_KEYWORDS = ['java', 'python', 'sql', 'excel', 'aws', 'azure', 'cisco', 'wan', 
                                'cybersecurity', 'pmp', 'agile', 'r', 'docker', 'jenkins', 
                                'windows', 'linux', 'ui/ux', 'database management', 'oracle',
                                'html', 'css', 'javascript', 'react', 'tensorflow', 'ai/ml',
                                'audit', 'risk management', 'firewall', 'vpn', 'security',
                                'manual testing', 'bug tracking', 'architecture', 'consulting',
                                'it strategy', 'business analysis', 'requirements', 'troubleshooting',
                                'customer service', 'devsecops', 'ci/cd', 'etl', 'big data',
                                'training', 'it education', 'cloud security', 'encryption',
                                'procurement', 'vendor management', 'user research', 'ux design',
                                'blockchain', 'solidity', 'risk assessment', 'compliance']
        def extract_keywords_from_sentences(text_data: str) -> List[str]:
            if pd.isna(text_data) or not isinstance(text_data, str): return []
            found_skills = set()
            text_data_lower = text_data.lower()
            for keyword in COMMON_TECH_KEYWORDS:
                if keyword in text_data_lower: found_skills.add(keyword.capitalize())
            return list(found_skills)
        temp_df['parsed_job_keywords'] = temp_df[CSV_SKILLS_COLUMN_STANDARDIZED].apply(extract_keywords_from_sentences)

        column_mappings = {
            'job_id': 'id', 'role': 'title', 'job_description': 'description',
            'company': 'company', 
            'parsed_job_keywords': 'requiredSkills',
            'parsed_experience_level': 'experience_level_required'
        }
        actual_mappings = {k: v for k, v in column_mappings.items() if k in temp_df.columns}
        temp_df.rename(columns=actual_mappings, inplace=True)
        
        if 'id' not in temp_df.columns:
            print("WARNING: 'id' column (from 'job_id') not found, generating new UUIDs.")
            temp_df['id'] = [uuid4().hex for _ in range(len(temp_df))] # Corrected: uuid4
        else:
            temp_df['id'] = temp_df['id'].astype(str)
        
        for field in ['title', 'company', 'description', 'requiredSkills', 'experience_level_required']:
            if field not in temp_df.columns:
                default_value = [] if field == 'requiredSkills' else \
                                'N/A' if field == 'company' else \
                                '' if field == 'description' else \
                                'Unknown' if field == 'experience_level_required' else \
                                'N/A' if field == 'title' else None
                temp_df[field] = default_value
        
        required_model_fields = ['id', 'title', 'company', 'description', 'requiredSkills', 'experience_level_required']
        final_columns = [col for col in required_model_fields if col in temp_df.columns]
        
        cleaned_df = temp_df[final_columns].copy()
        cleaned_df.dropna(subset=['id', 'title', 'description'], inplace=True) 

        if cleaned_df.empty:
            print("WARNING: DataFrame is empty after cleaning/filtering.")
            return None
        
        cleaned_df['title_safe'] = cleaned_df['title'].fillna('').astype(str)
        cleaned_df['skills_sentences_safe'] = temp_df.get(CSV_SKILLS_COLUMN_STANDARDIZED, pd.Series([''] * len(cleaned_df))).fillna('').astype(str)
        cleaned_df['description_safe'] = cleaned_df['description'].fillna('').astype(str)
        cleaned_df['text_for_embedding'] = cleaned_df['title_safe'] + ". " + cleaned_df['skills_sentences_safe'] + ". " + cleaned_df['description_safe']
        
        print(f"Successfully cleaned and prepared {len(cleaned_df)} IT job postings.")
        return cleaned_df

    except Exception as e:
        print(f"ERROR: Failed to load or process IT job dataset from CSV: {e}")
        import traceback; traceback.print_exc()
        return None

# --- Load ML Model and Data (with Caching) ---
# ... (Same model loading and embedding caching logic as before) ...
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

class Job(BaseModel):
    id: str; title: str
    company: Optional[str] = None; 
    description: Optional[str] = ""
    requiredSkills: List[str] = Field(default_factory=list)
    experience_level_required: Optional[str] = None 
class MatchedJob(Job): matchScore: float = Field(..., example=0.75)
class UserProfileResponse(BaseModel):
    uid: str; email: Optional[str] = None; displayName: Optional[str] = None
    photoURL: Optional[str] = None; createdAt: Optional[datetime] = None
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)

app = FastAPI()
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
    if total_years < 1: return "Entry-Level"
    elif total_years < 3: return "Junior"
    elif total_years < 7: return "Mid-Level"
    else: return "Senior"

def calculate_experience_level_match_score(user_level_str: str, job_level_str: Optional[str]) -> float:
    if not job_level_str or pd.isna(job_level_str) or job_level_str.lower().strip() in ['n/a', 'unknown', '']: return 0.5 
    user_level_norm = user_level_str.lower().replace('-', '').replace(' ', ''); job_level_norm = job_level_str.lower().replace('-', '').replace(' ', '')
    level_map = {"entrylevel": 0, "junior": 1, "midlevel": 2, "senior": 3}
    user_val = level_map.get(user_level_norm, -1); job_val = level_map.get(job_level_norm, -1)
    if user_val == -1 or job_val == -1: return 0.25 
    if user_val == job_val: return 1.0
    elif user_val > job_val: # Overqualified
        diff = user_val - job_val
        if diff == 1: return 0.7 
        elif diff == 2: return 0.4
        else: return 0.2
    elif job_val == user_val + 1: return 0.6 
    elif job_val == user_val + 2: return 0.2
    else: return 0.05

# --- ADDED THIS FUNCTION ---
def calculate_education_semantic_score(
    user_education_list: List[dict], 
    job_text_embedding: torch.Tensor, # Embedding of (job_title + job_skills_text + job_description)
    model: Optional[SentenceTransformer]
) -> float:
    if not user_education_list or model is None or job_text_embedding is None:
        return 0.0

    max_edu_score = 0.0
    for edu_item in user_education_list:
        degree_text = edu_item.get('degree', '').strip()
        school_text = edu_item.get('school', '').strip()
        # Combine degree and school for a richer education text
        education_entry_text = f"{degree_text} from {school_text}".strip()
        if not education_entry_text or education_entry_text == "from": # Handle empty strings
            education_entry_text = degree_text or school_text # Use whichever is available
        
        if education_entry_text:
            try:
                edu_embedding = model.encode(education_entry_text, convert_to_tensor=True)
                if len(edu_embedding.shape) == 1: edu_embedding = edu_embedding.unsqueeze(0)
                
                current_job_embedding_expanded = job_text_embedding
                if len(job_text_embedding.shape) == 1: # Should be 1D if it's a single job's embedding
                    current_job_embedding_expanded = job_text_embedding.unsqueeze(0)
                
                if not (torch.isnan(edu_embedding).any() or torch.isnan(current_job_embedding_expanded).any()):
                    score_tensor = util.cos_sim(edu_embedding, current_job_embedding_expanded)
                    score = float(score_tensor[0][0].item())
                    if score > max_edu_score:
                        max_edu_score = score
            except Exception as e:
                print(f"[WARN] Error embedding education item '{education_entry_text}': {e}")
    return max_edu_score
# --- END ADDED FUNCTION ---

@app.get("/")
async def read_root(): return {"message": "Resume Analyzer Backend is running!"}

@app.get("/api/profile", response_model=UserProfileResponse)
# ... (get_user_profile endpoint - same as before)
async def get_user_profile(current_user: Annotated[dict, Depends(get_current_user)]):
    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable")
    user_uid = current_user.get("uid")
    try:
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()
        if user_doc.exists:
            profile_data = user_doc.to_dict()
            profile_data["uid"] = user_uid 
            return UserProfileResponse(**profile_data)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found")
    except Exception as e:
        print(f"Error fetching profile for {user_uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not fetch profile")

@app.put("/api/profile/skills", response_model=SkillsUpdateRequest)
# ... (update_user_skills endpoint - same as before)
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
# ... (update_user_education endpoint - same as before)
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
# ... (update_user_experience endpoint - same as before)
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

@app.get("/api/jobs/match", response_model=List[MatchedJob])
async def get_job_matches(
    current_user: Annotated[dict, Depends(get_current_user)],
    min_score_threshold: float = 0.6,
    semantic_profile_weight: float = 0.3, # Adjusted weights
    keyword_weight: float = 0.3,        
    experience_level_weight: float = 0.25, 
    education_semantic_weight: float = 0.15,
    education_presence_bonus: float = 0.03 
):
    user_uid = current_user.get("uid") # ... (logging and checks) ...
    print(f"\n=== Starting HYBRID Job Match (New Dataset, EduSem) for User: {user_uid} ===")
    print(f"Weights: SemProfile={semantic_profile_weight}, KeySkill={keyword_weight}, ExpLvl={experience_level_weight}, EduSem={education_semantic_weight}, EduPresBonus={education_presence_bonus}")

    if db is None: raise HTTPException(status_code=503, detail="Firestore service unavailable (db)")
    if jobs_df is None or jobs_df.empty: print("[ERROR] Job DF not loaded."); return []
    if sentence_model is None: raise HTTPException(status_code=503, detail="Matching model unavailable")
    if job_embeddings is None or len(job_embeddings) != len(jobs_df):
         raise HTTPException(status_code=503, detail="Job embedding data unavailable")

    try:
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()
        if not user_doc.exists: raise HTTPException(404, "User profile not found")
        user_profile_data = user_doc.to_dict()
        
        user_skills_list = user_profile_data.get("skills", [])
        user_experience_list_raw = user_profile_data.get("experience", [])
        user_education_list_raw = user_profile_data.get("education", [])
        
        user_total_years_exp = get_user_total_experience_years(user_experience_list_raw)
        user_experience_level_cat = categorize_user_experience(user_total_years_exp)
        
        print(f"--- USER PROFILE FOR MATCHING (User UID: {user_uid}) ---")
        print(f"User Skills for Keyword: {user_skills_list}")
        print(f"User Total Years Exp: {user_total_years_exp:.2f}, Categorized Level: {user_experience_level_cat}")
        print(f"User Education Raw: {user_education_list_raw}")


        # User text for MAIN semantic embedding (Skills + Experience)
        user_profile_text_parts_main = []
        if user_skills_list: user_profile_text_parts_main.append("Skills: " + ", ".join(user_skills_list) + ".")
        if user_experience_list_raw:
            exp_texts = [f"Worked as {exp.get('title', '')} at {exp.get('company', '')}. Responsibilities: {exp.get('description', '')}" 
                         for exp in user_experience_list_raw if exp.get('title') and exp.get('company')]
            if exp_texts: user_profile_text_parts_main.append("Experience: " + " ".join(exp_texts))
        
        user_text_for_main_embedding = " ".join(user_profile_text_parts_main).strip()
        if not user_text_for_main_embedding: user_text_for_main_embedding = "General professional profile." # Default
        
        print(f"User Text for Main Embedding (Skills & Exp): '{user_text_for_main_embedding}'")
        
        user_main_embedding = sentence_model.encode(user_text_for_main_embedding, convert_to_tensor=True)
        if len(user_main_embedding.shape) == 1: user_main_embedding = user_main_embedding.unsqueeze(0)
        print(f"------------------------------------")

    except Exception as e:
        print(f"ERROR processing user profile for {user_uid}: {e}"); import traceback; traceback.print_exc()
        raise HTTPException(500, "Could not process user profile")

    matched_jobs_output = []
    try:
        all_main_semantic_scores_np = None
        if user_main_embedding is not None and not torch.isnan(user_main_embedding).any():
            cosine_scores_tensor = util.cos_sim(user_main_embedding, job_embeddings)[0] 
            all_main_semantic_scores_np = cosine_scores_tensor.cpu().numpy()
        else:
            print("[WARN] User main embedding is None or NaN, main semantic scores will be 0.")

        for idx, job_row in jobs_df.iterrows():
            # a. Main Semantic Score (User Skills+Exp vs Combined Job Text)
            main_semantic_score = float(all_main_semantic_scores_np[idx]) if all_main_semantic_scores_np is not None and idx < len(all_main_semantic_scores_np) else 0.0
            
            # b. Keyword Skill Score
            job_req_skills_list = job_row.get('requiredSkills', [])
            keyword_s = calculate_keyword_match_score(user_skills_list, job_req_skills_list)
            
            # c. Experience Level Score
            job_exp_level_req_str = job_row.get('experience_level_required')
            experience_s = calculate_experience_level_match_score(user_experience_level_cat, job_exp_level_req_str)

            # d. Education Semantic Score
            current_job_full_text_embedding = job_embeddings[idx] # This is embedding of (title + skills_sentences + description)
            education_semantic_s = calculate_education_semantic_score(user_education_list_raw, current_job_full_text_embedding, sentence_model)
            
            # e. Education Presence Bonus
            edu_presence_b = education_presence_bonus if user_education_list_raw else 0.0
            
            final_score = (semantic_profile_weight * main_semantic_score) + \
                          (keyword_weight * keyword_s) + \
                          (experience_level_weight * experience_s) + \
                          (education_semantic_weight * education_semantic_s) + \
                          edu_presence_b
            final_score = min(final_score, 1.0) 

            # --- Debugging for a specific job title ---
            target_job_title_debug = "Data Analyst" # Change this to a title you are investigating
            current_job_title_debug = job_row.get('title', 'N/A')
            if target_job_title_debug.lower() in current_job_title_debug.lower(): # Or check by ID
                print(f"\n--- DEBUG JOB: {current_job_title_debug} (ID: {job_row.get('id')}) ---")
                print(f"Job Text for Embedding: {job_row.get('text_for_embedding', 'N/A')[:200]}...") # Print start of job text
                print(f"Scores (pre-weight): MainSem={main_semantic_score:.3f}, Keyword={keyword_s:.3f}, ExpLvl={experience_s:.3f}, EduSem={education_semantic_s:.3f}")
                print(f"EduPresenceBonus: {edu_presence_b:.3f}")
                print(f"Weighted: MainSem={(semantic_profile_weight * main_semantic_score):.3f}, Key={(keyword_weight * keyword_s):.3f}, ExpLvl={(experience_level_weight * experience_s):.3f}, EduSem={(education_semantic_weight * education_semantic_s):.3f}")
                print(f"Final Score: {final_score:.3f}")
                print(f"-------------------------------------------\n")
                
            if final_score >= min_score_threshold:
                try:
                    job_data_dict = job_row.to_dict()
                    job_instance_data = {
                        "id": str(job_data_dict.get('id', '')),
                        "title": job_data_dict.get('title', 'N/A'), 
                        "company": job_data_dict.get('company', None),
                        "description": job_data_dict.get('description', ""), 
                        "requiredSkills": job_data_dict.get('requiredSkills', []), 
                        "experience_level_required": job_data_dict.get('experience_level_required', None),
                        "matchScore": final_score
                    }
                    matched_jobs_output.append(MatchedJob(**job_instance_data))
                except Exception as val_err:
                    print(f"[ERROR] Skipping job index {idx} (ID: {job_data_dict.get('id', 'UNKNOWN')}) due to Pydantic validation: {val_err}")

    except Exception as e:
        print(f"ERROR during job processing loop: {e}"); import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error calculating job matches")

    matched_jobs_output.sort(key=lambda job: job.matchScore, reverse=True)
    print(f"=== HYBRID Job Match (New Dataset, EduSem) Finished. Found {len(matched_jobs_output)} matches. ===")
    return matched_jobs_output

