from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

origins = [
    "http://localhost:3000", # Your frontend origin
    # Add any other origins if needed (e.g., your deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

@app.get("/")
async def read_root():
    return " Backend is running!"