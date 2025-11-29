import os, json
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sqlalchemy.orm import Session
from .settings import get_settings
from .db import get_db
from .models import GoogleToken
from .security import get_current_user_id

router = APIRouter(prefix="/google", tags=["google"])

def _client_config():
    s = get_settings()
    redirect_uri = f"{s.BASE_URL.rstrip('/')}/google/callback"
    return {
        "web": {
            "client_id": s.GOOGLE_API_CLIENT_ID,
            "client_secret": s.GOOGLE_API_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }

@router.get("/auth")
def start_auth(request: Request):
    user_id = get_current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    # inside start_auth()
    s = get_settings()
    if not s.GOOGLE_API_CLIENT_ID or not s.GOOGLE_API_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Missing GOOGLE_API_CLIENT_ID / GOOGLE_API_CLIENT_SECRET")
    redirect_uri = f"{s.BASE_URL.rstrip('/')}/google/callback"

    flow = Flow.from_client_config(
        _client_config(),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
        redirect_uri=redirect_uri,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    request.session['google_oauth_state'] = state
    return RedirectResponse(auth_url)

@router.get("/callback")
def oauth_callback(request: Request, db: Session = Depends(get_db), state: str | None = None, code: str | None = None):
    user_id = get_current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    # inside start_auth()
    s = get_settings()
    if not s.GOOGLE_API_CLIENT_ID or not s.GOOGLE_API_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Missing GOOGLE_API_CLIENT_ID / GOOGLE_API_CLIENT_SECRET")
    redirect_uri = f"{s.BASE_URL.rstrip('/')}/google/callback"

    flow = Flow.from_client_config(
        _client_config(),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
        redirect_uri=redirect_uri,
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    token_json = creds.to_json()
    existing = db.query(GoogleToken).filter_by(user_id=user_id).first()
    if existing:
        existing.token_json = token_json
    else:
        db.add(GoogleToken(user_id=user_id, token_json=token_json))
    db.commit()
    return RedirectResponse(url="/dashboard?google=connected")


@router.get("/disconnect")
def disconnect_google(request: Request, db: Session = Depends(get_db)):
    """Disconnect Google Drive by removing the stored token."""
    user_id = get_current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    
    # Delete the Google token from the database
    existing = db.query(GoogleToken).filter_by(user_id=user_id).first()
    if existing:
        db.delete(existing)
        db.commit()
        print(f"[Google OAuth] User {user_id} disconnected Google Drive")
    
    # Also remove the cached token file if it exists
    try:
        token_path = os.path.join("data", "google_tokens", f"user_{user_id}.json")
        if os.path.exists(token_path):
            os.remove(token_path)
            print(f"[Google OAuth] Removed cached token file: {token_path}")
    except Exception as e:
        print(f"[Google OAuth] Warning: Could not remove token file: {e}")
    
    return RedirectResponse(url="/dashboard?google=disconnected")
