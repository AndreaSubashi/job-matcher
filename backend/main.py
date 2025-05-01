
from fastapi import FastAPI, Depends, HTTPException, status, Header 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field 
from typing import List, Annotated, Optional 
from datetime import datetime #added because of an error when trying to interpret date/time as a string instead of date/time
from uuid import UUID, uuid4

import firebase_admin
from firebase_admin import credentials, auth, firestore # Import auth and firestore

# --- Firebase Admin SDK Initialization ---
try:
    # Use the downloaded service account key
    cred = credentials.Certificate("firebase-service-account.json")
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")

# Get Firestore client
try:
    db = firestore.client()
    print("Firestore client obtained successfully.")
except Exception as e:
    print(f"Error obtaining Firestore client: {e}")
    db = None # Set db to None if initialization fails
# --- End Firebase Admin SDK Initialization ---


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
    company: str
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
    experience: List[ExperienceItem] = Field(default_factory=list)


def calculate_match_score(user_skills: List[str], job_skills: List[str]) -> float:
    """
    Calculates a simple match score based on skill overlap.
    Score = (Number of matching skills) / (Total number of required job skills)
    Returns a float between 0.0 and 1.0.
    """

    # Use sets for efficient intersection finding, case-insensitive comparison
    user_skills_set = set(skill.lower() for skill in user_skills)
    job_skills_set = set(skill.lower() for skill in job_skills)

    matching_skills = user_skills_set.intersection(job_skills_set)
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

    if not job_skills:
        return 0.0 # Or 1.0 if no skills required means perfect match? Decide business logic. Let's say 0.
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
@app.get("/api/jobs/match", response_model=List[MatchedJob]) # Return a list of MatchedJob
async def get_job_matches(
    current_user: Annotated[dict, Depends(get_current_user)],
    min_score_threshold: float = 0.25 # Optional query parameter for minimum score
):
    """
    Fetches jobs, matches them against the authenticated user's skills,
    and returns a ranked list of jobs exceeding the score threshold.
    """
    if db is None:
         raise HTTPException(status_code=503, detail="Firestore service unavailable")

    user_uid = current_user.get("uid")
    print(f"\n=== Starting Job Match for User: {user_uid} ===") # Log start
    if not user_uid:
         raise HTTPException(status_code=400, detail="User ID not found in token")

    matched_jobs = []

    try:
        # 1. Fetch User Profile (specifically skills)
        user_doc_ref = db.collection("users").document(user_uid)
        user_doc = user_doc_ref.get()
        user_profile_data = user_doc.to_dict()
        user_skills = user_profile_data.get("skills", [])
        print(f"[DEBUG] User Skills Fetched: {user_skills}")
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User profile not found, cannot match jobs.")

        if not user_skills:
             print(f"User {user_uid} has no skills listed, returning empty match list.")
             return [] # No skills to match with

        # 2. Fetch All Job Listings
        print("[DEBUG] Attempting to fetch 'jobs' collection...")
        jobs_ref = db.collection("jobs")
        job_docs = jobs_ref.stream() # Use stream() for potentially large collections
        print(f"[DEBUG] Got job stream object: {job_docs}")
        job_count = 0 # Counter for logging

        # 3. Calculate Score for Each Job
        for job_doc in job_docs:
            print("[DEBUG] Entering loop to process job documents...")
            job_count += 1
            job_data = job_doc.to_dict()
            job_data["id"] = job_doc.id # Add the document ID to the job data

            # Basic validation/defaults
            job_data.setdefault("requiredSkills", [])
            job_data.setdefault("title", "N/A")
            job_data.setdefault("company", "N/A")

            required_skills = job_data.get("requiredSkills", [])
            # --- ADD DEBUG LOGGING ---
            print(f"[DEBUG] Job {job_count} ({job_doc.id}): Title='{job_data.get('title', 'N/A')}', Required Skills={required_skills}")
            # --- END DEBUG LOGGING ---

            score = calculate_match_score(user_skills, required_skills)

            # 4. Filter by Threshold
            print(f"[DEBUG] Job {job_doc.id} Score: {score:.2f} (Threshold: {min_score_threshold})")
            if score >= min_score_threshold:
                 # Create a MatchedJob object (inherits Job fields, adds score)
                 # Ensure job_data has all fields required by Job model before unpacking
                 print(f"[DEBUG] Job {job_doc.id} PASSED threshold.") # Log pass
                 try:
                    matched_job_data = {
                        **job_data, # Spread job data
                        "matchScore": score
                    }
                    # Validate against MatchedJob model before adding
                    matched_jobs.append(MatchedJob(**matched_job_data))
                 except Exception as validation_error: # Catch potential Pydantic validation errors
                     print(f"Skipping job {job_doc.id} due to validation error: {validation_error}")
                     print(f"Job data was: {job_data}")

        # Add a check *after* the loop
        if job_count == 0:
             print("[DEBUG] Loop finished without processing any jobs (check if 'jobs' collection exists and has documents).") #<-- ADD THIS LINE

        # 5. Rank Results (Higher score first)
        matched_jobs.sort(key=lambda job: job.matchScore, reverse=True)

        print(f"Found {len(matched_jobs)} job matches for user {user_uid} with threshold {min_score_threshold}")
        return matched_jobs

    except Exception as e:
        print(f"Error during job matching for {user_uid}: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not perform job matching")
    



