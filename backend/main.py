
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

class UserProfileResponse(BaseModel):
    uid: str
    email: Optional[str] = None
    displayName: Optional[str] = None
    photoURL: Optional[str] = None
    createdAt: Optional[datetime] = None # here
    skills: List[str] = Field(default_factory=list) # default_factory for empty lists
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)

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
# --- Add job matching later ---