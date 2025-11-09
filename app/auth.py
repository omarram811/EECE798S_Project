from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from .db import get_db
from .models import User
from .security import hash_password, verify_password, set_session_cookie, get_current_user_id, clear_session_cookie

router = APIRouter(tags=["auth"])

@router.get("/logout")
def logout():
    resp = RedirectResponse("/")
    resp.delete_cookie("session")
    clear_session_cookie(resp)
    return resp

@router.get("/register")
def register_form():
    return RedirectResponse("/?register=1")

@router.post("/register")
def register(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(status_code=400, detail="Email exists")
    u = User(email=email, password_hash=hash_password(password))
    db.add(u)
    db.commit()
    resp = RedirectResponse("/dashboard", status_code=302)
    set_session_cookie(resp, u.id)
    return resp

@router.post("/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    u = db.query(User).filter_by(email=email).first()
    if not u or not verify_password(password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    resp = RedirectResponse("/dashboard", status_code=302)
    set_session_cookie(resp, u.id)
    return resp
