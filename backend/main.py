import os
import pandas as pd
import ast # For parsing stringified lists like '["skill1", "skill2"]'
from fastapi import FastAPI, Depends, HTTPException, status, Header 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field 
from typing import List, Annotated, Optional 
from datetime import datetime #added because of an error when trying to interpret date/time as a string instead of date/time
from uuid import UUID, uuid4
from dotenv import load_dotenv 
from sentence_transformers import SentenceTransformer, util # Import sentence-transformers
import torch # Import torch


import firebase_admin
from firebase_admin import credentials, auth, firestore # Import auth and firestore

# --- Load Environment Variables ---
load_dotenv()

# --- Firebase Admin SDK Initialization ---
try:
    # Use the downloaded service account key
    cred = credentials.Certificate("firebase-service-account.json")
    # Avoid re-initializing if already done (useful for some environments)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    else:
        print("Firebase Admin SDK already initialized.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")

# Get Firestore client
try:
    db = firestore.client()
    print("Firestore client obtained successfully.")
except Exception as e:
    print(f"Error obtaining Firestore client: {e}")
# --- End Firebase Admin SDK Initialization ---
# --- Global variable to hold job data ---
jobs_df = None
# --- !!! IMPORTANT: SET CORRECT PATH TO YOUR DOWNLOADED CSV !!! ---
JOBS_DATA_PATH = "data/job_skill_set.csv"


# --- Load ML Model on Startup ---
try:
    # Choose a pre-trained model. 'all-MiniLM-L6-v2' is fast and good quality.
    # Other options: 'msmarco-distilbert-base-v4', 'all-mpnet-base-v2' (slower but potentially better)
    MODEL_NAME = 'all-MiniLM-L6-v2'
    print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    sentence_model = SentenceTransformer(MODEL_NAME)
    # You can check the embedding dimension:
    # print(f"Model embedding dimension: {sentence_model.get_sentence_embedding_dimension()}")
    print("Sentence Transformer model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load Sentence Transformer model: {e}")
    # Decide if the app should fail or continue without ML matching
    sentence_model = None

# --- Load Job Data on Startup ---
job_embeddings = None # Variable to hold job embeddings
try:
    print(f"Loading job dataset from: {JOBS_DATA_PATH}...")
    temp_df = pd.read_csv(JOBS_DATA_PATH)
    print(f"Initial rows loaded: {len(temp_df)}")

    # --- Data Cleaning / Preparation ---
    # 1. Handle missing descriptions
    temp_df['job_description'].fillna('', inplace=True)
    temp_df['job_title'].fillna('N/A', inplace=True) # Handle missing titles too

    # 2. Handle skills column - !!! VERY IMPORTANT: INSPECT YOUR CSV !!!
    # Check the actual column name for skills (e.g., 'job_skill_set') and adjust below.
    # Check how the skills are formatted (e.g., '["skill1","skill2"]' or 'skill1, skill2')
    # Adjust the parsing logic accordingly. Using ast.literal_eval for '["skill1","skill2"]' format.
    SKILLS_COLUMN_NAME = 'job_skill_set' # <-- ADJUST IF YOUR COLUMN NAME IS DIFFERENT

    if SKILLS_COLUMN_NAME not in temp_df.columns:
        print(f"WARNING: Skills column '{SKILLS_COLUMN_NAME}' not found. Adding empty skills list.")
        temp_df['requiredSkills'] = pd.Series([[] for _ in range(len(temp_df))])
    else:
        print(f"Parsing skills from column: '{SKILLS_COLUMN_NAME}'...")
        def parse_skill_list(skill_str):
            if pd.isna(skill_str) or not isinstance(skill_str, str) or not skill_str.startswith('['):
                return []
            try:
                # Safely evaluate the string literal as a Python list
                skills = ast.literal_eval(skill_str)
                # Ensure all items are strings and strip whitespace
                return [str(s).strip() for s in skills if str(s).strip()]
            except (ValueError, SyntaxError, TypeError) as parse_error:
                # print(f"Warning: Could not parse skill string: {skill_str} | Error: {parse_error}") # Optional: Log parsing errors
                return [] # Return empty list if parsing fails

        temp_df['requiredSkills'] = temp_df[SKILLS_COLUMN_NAME].apply(parse_skill_list)
        print(f"Finished parsing skills. Example parsed skills: {temp_df['requiredSkills'].iloc[0] if not temp_df.empty else 'N/A'}")


    # 3. Rename columns to match Pydantic Job model
    #    Add/remove mappings based on your CSV columns and Job model fields
    column_mappings = {
        'job_id': 'id',             # Adjust CSV column name if different
        'job_title': 'title',
        'job_description': 'description',
        # Add mappings for company, location if they exist in CSV
        # 'Company Name': 'company',
        # 'Location': 'location',
    }
    # Only rename columns that actually exist in the DataFrame
    actual_mappings = {k: v for k, v in column_mappings.items() if k in temp_df.columns}
    temp_df.rename(columns=actual_mappings, inplace=True)

    # 4. Ensure required model columns exist, add defaults if necessary
    if 'id' not in temp_df.columns:
        print("WARNING: 'id' column not found after mapping, generating new IDs.")
        temp_df['id'] = [uuid4().hex for _ in range(len(temp_df))]
    else:
         # Ensure existing IDs are strings
         temp_df['id'] = temp_df['id'].astype(str)

    if 'company' not in temp_df.columns: temp_df['company'] = 'N/A'
    if 'location' not in temp_df.columns: temp_df['location'] = None

    # 5. Select final columns needed for the Job model + matching
    required_model_fields = ['id', 'title', 'company', 'location', 'description', 'requiredSkills']
    # Keep only columns that actually exist after potential additions/renames
    final_columns = [col for col in required_model_fields if col in temp_df.columns]
    jobs_df = temp_df[final_columns].copy()

    # 6. Drop rows where essential info is missing AFTER selection
    jobs_df.dropna(subset=['id', 'title'], inplace=True)
    # Optional: Fill NaN in list columns explicitly if needed, though default_factory handles it too
    # jobs_df['requiredSkills'] = jobs_df['requiredSkills'].apply(lambda x: x if isinstance(x, list) else [])


    print(f"Successfully loaded and prepared {len(jobs_df)} job postings into DataFrame.")
    # print(jobs_df.info()) # Print DataFrame info (columns, types, memory)

    if not temp_df.empty:
        # Select final columns (ensure 'description' is included)
        required_model_fields = ['id', 'title', 'company', 'location', 'description', 'requiredSkills']
        final_columns = [col for col in required_model_fields if col in temp_df.columns]
        jobs_df = temp_df[final_columns].copy()
        jobs_df.dropna(subset=['id', 'title', 'description'], inplace=True) # Ensure description exists for embedding
        jobs_df['id'] = jobs_df['id'].astype(str)
        print(f"Successfully loaded and prepared {len(jobs_df)} job postings into DataFrame.")

        # --- Pre-compute Job Embeddings (if model loaded successfully) ---
        if sentence_model is not None and not jobs_df.empty:
            print(f"Calculating embeddings for {len(jobs_df)} job descriptions...")
            # Combine title and description for better context (optional)
            # jobs_df['text_for_embedding'] = jobs_df['title'] + " " + jobs_df['description']
            # Or just use description:
            job_descriptions = jobs_df['description'].tolist()
            # Calculate embeddings (this can take time/RAM depending on dataset size and model)
            job_embeddings = sentence_model.encode(job_descriptions, convert_to_tensor=True, show_progress_bar=True)
            print(f"Job embeddings calculated. Shape: {job_embeddings.shape}")
            # Store embeddings (optional, can access directly from jobs_df index)
            # You could add them as a column, but keeping separate might be cleaner
            # jobs_df['embedding'] = job_embeddings.tolist() # Convert tensor rows to lists for DataFrame storage (can be inefficient)

    else:
        print("WARNING: DataFrame is empty after cleaning.")
        jobs_df = None

except FileNotFoundError:
    print(f"ERROR: Job dataset file not found at {JOBS_DATA_PATH}. Job matching will use Firestore (if implemented) or fail.")
    jobs_df = None # Ensure it's None if file not found
except Exception as e:
    print(f"ERROR: Failed to load or process job dataset from CSV: {e}")
    # import traceback # Uncomment for detailed trace
    # traceback.print_exc()
    jobs_df = None

app = FastAPI()

# Configure CORS 
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication Dependency ---
async def get_current_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    """
    Dependency function to verify Firebase ID token and get user data.
    Expects 'Authorization: Bearer <token>' header.
    """
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Bearer <token>'",
        )

    id_token = parts[1]
    try:
        # Verify the ID token while checking if the token is revoked.
        decoded_token = auth.verify_id_token(id_token, check_revoked=True)
        # You can return the whole decoded token or just the UID
        # print("Decoded token:", decoded_token) # For debugging
        return decoded_token # Contains 'uid', 'email', etc.
    except auth.ExpiredIdTokenError:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ID token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.RevokedIdTokenError:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ID token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ID token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        print(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not verify token",
        )
# --- End Authentication Dependency ---

# --- Pydantic Models ---
class SkillsUpdateRequest(BaseModel):
    skills: List[str] = Field(..., example=["Python", "React", "SQL"])

class EducationItem(BaseModel):
    id: UUID = Field(default_factory=uuid4) # Automatically generate UUID if not provided
    degree: str = Field(..., example="Bachelor of Science")
    school: str = Field(..., example="University of Example")
    startYear: Optional[int] = Field(None, example=2018)
    endYear: Optional[int] = Field(None, example=2022)

class EducationUpdateRequest(BaseModel):
    # Expects a list of education items. The frontend will send the whole list.
    education: List[EducationItem]

class ExperienceItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., example="Software Engineer")
    company: str = Field(..., example="Tech Solutions Inc.")
    # Using strings for dates - easier for typical YYYY-MM form inputs
    startDate: Optional[str] = Field(None, example="2020-01") # Or use date type: Optional[date] = None
    endDate: Optional[str] = Field(None, example="2022-12") # Or use date type: Optional[date] = None; Null/empty means 'Present'
    location: Optional[str] = Field(None, example="Remote")
    description: Optional[str] = Field(None, example="Developed features for...")

# NEW: Model for the request body when updating the whole experience list
class ExperienceUpdateRequest(BaseModel):
    experience: List[ExperienceItem]

class Job(BaseModel):
    id: str # Document ID from Firestore
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    requiredSkills: List[str] = Field(default_factory=list)
    # Add other fields like url, datePosted later if needed

# NEW: Model representing a Job returned after matching, includes a score
class MatchedJob(Job): # Inherits fields from Job
    matchScore: float = Field(..., example=0.75) # Add the match score (e.g., 0.0 to 1.0)


class UserProfileResponse(BaseModel):
    uid: str
    email: Optional[str] = None
    displayName: Optional[str] = None
    photoURL: Optional[str] = None
    createdAt: Optional[datetime] = None # here
    skills: List[str] = Field(default_factory=list) # default_factory for empty lists
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[EducationItem] = Field(default_factory=list)


# --- Helper Functions ---
def calculate_match_score(user_skills: List[str], job_skills: List[str]) -> float:
    """
    Calculates a simple match score based on skill overlap.
    Score = (Number of matching skills) / (Total number of required job skills)
    Returns a float between 0.0 and 1.0.
    """
    # --- !!! MOVED THIS CHECK TO THE TOP !!! ---
    if not job_skills:
        print("[DEBUG] calculate_match_score: Job has no required skills. Score: 0.0")
        return 0.0 # No skills required, or handle as 1.0 if preferred

    # Use sets for efficient intersection finding, case-insensitive comparison
    user_skills_set = set(skill.lower() for skill in user_skills)
    job_skills_set = set(skill.lower() for skill in job_skills)

    matching_skills = user_skills_set.intersection(job_skills_set)
    # No need to check len(job_skills_set) again due to the check above
    score = len(matching_skills) / len(job_skills_set)

    # --- ADD DEBUG LOGGING ---
    print(f"--- Score Calculation ---")
    print(f"User Skills Set: {user_skills_set}")
    print(f"Job Skills Set: {job_skills_set}")
    print(f"Matching Skills: {matching_skills}")
    print(f"Score: {score:.2f}") # Format score
    print(f"-----------------------")
    # --- END DEBUG LOGGING ---

    return score
# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Resume Analyzer Backend is running!"}

# --- Profile Endpoints (Skills) ---
@app.get("/api/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: Annotated[dict, Depends(get_current_user)]
):
    """
    Fetches the complete profile for the authenticated user from Firestore.
    """
    if db is None:
         raise HTTPException(status_code=503, detail="Firestore service unavailable")

    user_uid = current_user.get("uid")
    if not user_uid:
         raise HTTPException(status_code=400, detail="User ID not found in token")

    try:
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()

        if user_doc.exists:
            profile_data = user_doc.to_dict()
            profile_data["uid"] = current_user.get("uid")
            return UserProfileResponse(**profile_data)
        else:
             #return 404
             print(f"Profile document not found for UID: {user_uid}")
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found")

    except Exception as e:
        print(f"Error fetching profile for {current_user.get('uid')}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not fetch profile")


@app.put("/api/profile/skills", response_model=SkillsUpdateRequest)
async def update_user_skills(
    skills_update: SkillsUpdateRequest,
    current_user: Annotated[dict, Depends(get_current_user)]
):
    """
    Updates the skills list for the authenticated user in Firestore.
    Replaces the existing skills list with the provided one.
    """
    if db is None:
         raise HTTPException(status_code=503, detail="Firestore service unavailable")

    user_uid = current_user.get("uid")
    if not user_uid:
         raise HTTPException(status_code=400, detail="User ID not found in token")

    try:
        user_doc_ref = db.collection("users").document(user_uid)
        # Use update() which can also create fields if the doc exists but the field doesn't
        # Using set(..., merge=True) would also work to only update specific fields
        user_doc_ref.update({"skills": skills_update.skills})
        print(f"Updated skills for UID: {user_uid}")
        return skills_update # Return the updated skills list
    except Exception as e:
        print(f"Error updating skills for {user_uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not update skills")
    
# --- Profile Endpoints (Education) ---
@app.put("/api/profile/education", response_model=EducationUpdateRequest)
async def update_user_education(
    education_update: EducationUpdateRequest, # Use the new request model
    current_user: Annotated[dict, Depends(get_current_user)]
):
    """
    Updates the education list for the authenticated user in Firestore.
    Replaces the existing education list with the provided one.
    """
    if db is None:
         raise HTTPException(status_code=503, detail="Firestore service unavailable")

    user_uid = current_user.get("uid")
    if not user_uid:
         raise HTTPException(status_code=400, detail="User ID not found in token")

    try:
        user_doc_ref = db.collection("users").document(user_uid)

        # Convert Pydantic models to dicts AND ensure ID is a string for Firestore
        education_list_for_firestore = []
        for item in education_update.education:
            item_dict = item.model_dump()
            # Explicitly convert UUID object to string hex representation
            item_dict['id'] = str(item.id) # <--- CHANGE IS HERE
            education_list_for_firestore.append(item_dict)

        # Update the 'education' field in Firestore with the list of dicts (ID as string)
        user_doc_ref.update({"education": education_list_for_firestore})

        print(f"Updated education for UID: {user_uid}")
        # Return the original Pydantic model - FastAPI handles converting UUID back for JSON if needed
        return education_update

    except Exception as e:
        # **CRITICAL:** Look at the terminal output for the *exact* error 'e' here!
        print(f"Error updating education for {user_uid}: {e}")
        # You might want to log the traceback too:
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not update education")

# --- Profile Endpoints (Experience) ---
@app.put("/api/profile/experience", response_model=ExperienceUpdateRequest)
async def update_user_experience(
    experience_update: ExperienceUpdateRequest, # Use the new request model
    current_user: Annotated[dict, Depends(get_current_user)]
):
    """
    Updates the experience list for the authenticated user in Firestore.
    Replaces the existing experience list with the provided one.
    """
    if db is None:
         raise HTTPException(status_code=503, detail="Firestore service unavailable")

    user_uid = current_user.get("uid")
    if not user_uid:
         raise HTTPException(status_code=400, detail="User ID not found in token")

    try:
        user_doc_ref = db.collection("users").document(user_uid)

        # Convert Pydantic models back to dicts, ensuring ID is string for Firestore
        experience_list_for_firestore = []
        for item in experience_update.experience:
            item_dict = item.model_dump()
            # Explicitly convert UUID object to string hex representation
            item_dict['id'] = str(item.id)
            # Ensure dates are strings or null (if using string type in model)
            # If using date type in model, Firestore handles date objects directly.
            # item_dict['startDate'] = item.startDate.isoformat() if item.startDate else None # Example if using date type
            # item_dict['endDate'] = item.endDate.isoformat() if item.endDate else None # Example if using date type
            experience_list_for_firestore.append(item_dict)

        # Update the 'experience' field in Firestore
        user_doc_ref.update({"experience": experience_list_for_firestore})

        print(f"Updated experience for UID: {user_uid}")
        # Return the data that was sent
        return experience_update

    except Exception as e:
        print(f"Error updating experience for {user_uid}: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback during debugging
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not update experience")

# --- Job Matching Endpoint ---
@app.get("/api/jobs/match", response_model=List[MatchedJob])
async def get_job_matches(
    current_user: Annotated[dict, Depends(get_current_user)],
    min_score_threshold: float = 0.3 # Adjust threshold for semantic scores (often higher)
):
    user_uid = current_user.get("uid")
    print(f"\n=== Starting SEMANTIC Job Match for User: {user_uid} ===")

    # --- Pre-computation Checks ---
    if jobs_df is None or jobs_df.empty:
        print("[ERROR] Job dataset (DataFrame) not loaded. Cannot match.")
        return []
    if sentence_model is None:
        print("[ERROR] Sentence Transformer model not loaded. Cannot perform semantic match.")
        raise HTTPException(status_code=503, detail="Matching service model unavailable")
    if job_embeddings is None or len(job_embeddings) != len(jobs_df):
         print("[ERROR] Job embeddings not pre-computed or mismatch length. Cannot perform semantic match.")
         # Optionally: Compute on-the-fly here if not pre-computed, but will be slow
         raise HTTPException(status_code=503, detail="Job embedding data unavailable")


    # 1. Fetch User Profile Skills
    try:
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()
        if not user_doc.exists: raise HTTPException(404, "User profile not found")
        user_profile_data = user_doc.to_dict()
        user_skills = user_profile_data.get("skills", [])
        print(f"[DEBUG] User Skills Fetched: {user_skills}")
        if not user_skills:
             print("[INFO] User has no skills. Cannot perform semantic match based on skills.")
             return []
        # Combine skills into a single string for embedding
        user_skills_text = ", ".join(user_skills)
        print(f"[DEBUG] User skills text for embedding: '{user_skills_text}'")

    except Exception as e:
        print(f"ERROR fetching user profile {user_uid}: {e}")
        raise HTTPException(500, "Could not fetch user profile")

    # 2. Calculate Similarity & Filter
    matched_jobs_indices = [] # Store tuples of (index, score)
    try:
        print("[DEBUG] Calculating user skill embedding...")
        user_embedding = sentence_model.encode(user_skills_text, convert_to_tensor=True)
        print("[DEBUG] Calculating cosine similarities...")

        # Compute cosine similarity between user embedding and all pre-computed job embeddings
        # util.cos_sim returns a tensor of shape [1, num_jobs]
        cosine_scores = util.cos_sim(user_embedding, job_embeddings)[0] # Get the first row

        # Convert to numpy array for easier handling
        cosine_scores_np = cosine_scores.cpu().numpy()

        print(f"[DEBUG] Calculated {len(cosine_scores_np)} scores.")
        # Find indices where score meets threshold
        for idx, score in enumerate(cosine_scores_np):
            score = float(score) # Convert from numpy float
            # print(f"[DEBUG] Job index {idx} Score: {score:.4f}") # Verbose logging
            if score >= min_score_threshold:
                matched_jobs_indices.append((idx, score))
                # print(f"[DEBUG] Job index {idx} PASSED threshold.")

    except Exception as e:
        print(f"ERROR during semantic similarity calculation: {e}")
        raise HTTPException(status_code=500, detail="Error calculating job matches")


    # 3. Format & Rank Results
    matched_jobs = []
    print(f"[DEBUG] Formatting {len(matched_jobs_indices)} potential matches...")
    for idx, score in matched_jobs_indices:
        try:
            job_row = jobs_df.iloc[idx] # Get job data using index
            job_data = job_row.to_dict()
            job_instance_data = {
                "id": str(job_data.get('id', '')),
                "title": job_data.get('title', 'N/A'),
                "company": job_data.get('company', None),
                "location": job_data.get('location', None),
                "description": job_data.get('description', ""),
                 # Ensure requiredSkills is a list from the DataFrame row
                "requiredSkills": job_data.get('requiredSkills', []) if isinstance(job_data.get('requiredSkills'), list) else [],
                "matchScore": score # Use the calculated cosine similarity score
            }
            matched_jobs.append(MatchedJob(**job_instance_data))
        except Exception as validation_error:
             print(f"[ERROR] Skipping job index {idx} (ID: {job_data.get('id', 'UNKNOWN') if 'job_data' in locals() else 'UNKNOWN'}) due to Pydantic validation error: {validation_error}")


    # Rank results (higher score first)
    matched_jobs.sort(key=lambda job: job.matchScore, reverse=True)

    print(f"=== SEMANTIC Job Match Finished. Found {len(matched_jobs)} matches. ===")
    return matched_jobs
    



