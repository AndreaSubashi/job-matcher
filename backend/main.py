# backend/main.py
from fastapi import FastAPI, Depends, HTTPException, status, Header # Added Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Import Pydantic BaseModel and Field
from typing import List, Annotated, Optional # Import List and Annotated
from datetime import datetime # <--- IMPORT THIS

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
    # Depending on your setup, you might want to exit or handle this differently
    # raise e # Re-raise if you want the app to fail on init error

# Get Firestore client
try:
    db = firestore.client()
    print("Firestore client obtained successfully.")
except Exception as e:
    print(f"Error obtaining Firestore client: {e}")
    db = None # Set db to None if initialization fails
# --- End Firebase Admin SDK Initialization ---


app = FastAPI()

# Configure CORS (as before)
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
        print(f"Token verification failed: {e}") # Log other errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not verify token",
        )
# --- End Authentication Dependency ---

# --- Pydantic Models ---
class SkillsUpdateRequest(BaseModel):
    skills: List[str] = Field(..., example=["Python", "React", "SQL"])

class UserProfileResponse(BaseModel):
    uid: str
    email: Optional[str] = None
    displayName: Optional[str] = None
    photoURL: Optional[str] = None
    createdAt: Optional[datetime] = None # Or use datetime
    skills: List[str] = Field(default_factory=list) # Use default_factory for empty lists
    education: List[dict] = Field(default_factory=list)
    experience: List[dict] = Field(default_factory=list)
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
            # Ensure default empty lists if fields are missing
            #profile_data.setdefault("skills", [])
            #profile_data.setdefault("education", [])
            #profile_data.setdefault("experience", [])
            # Add uid which might not be stored in the doc itself
            profile_data["uid"] = current_user.get("uid")
            return UserProfileResponse(**profile_data)
        else:
             # Optionally create a basic profile if it's missing but shouldn't be?
             # Or just return 404
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

# --- Add endpoints for education, experience, and job matching later ---