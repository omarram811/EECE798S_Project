import os
from fastapi import FastAPI, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from .db import Base, engine, get_db
from .models import User, Agent, Conversation, Message, GoogleToken
from .security import get_current_user_id
from .auth import router as auth_router
from .agents import router as agents_router
from .chat import router as chat_router
from .google_oauth import router as google_router
from .scheduler import start_scheduler

load_dotenv()
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Course TA Agent Studio")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY","dev"))
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(auth_router)
app.include_router(agents_router)
app.include_router(chat_router)
app.include_router(google_router)

@app.on_event("startup")
def _startup():
    start_scheduler()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    uid = get_current_user_id(request)
    return templates.TemplateResponse("home.html", {"request": request, "user_id": uid})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        return RedirectResponse("/")
    agents = db.query(Agent).filter_by(owner_id=uid).all()
    google_connected = db.query(GoogleToken).filter_by(user_id=uid).first() is not None
    return templates.TemplateResponse("dashboard.html", {"request": request, "agents": agents, "google_connected": google_connected})

@app.get("/a/{slug}", response_class=HTMLResponse)
def chat_page(slug: str, request: Request, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        return RedirectResponse("/")
    agent = db.query(Agent).filter_by(slug=slug).first()
    if not agent:
        raise HTTPException(status_code=404)
    conv = db.query(Conversation).filter_by(agent_id=agent.id, user_id=uid).order_by(Conversation.id.desc()).first()
    messages = []
    if conv:
        messages = db.query(Message).filter_by(conversation_id=conv.id).order_by(Message.id.asc()).all()
    return templates.TemplateResponse("chat.html", {"request": request, "agent": agent, "messages": messages})

# simple public redirect for sharing
@app.get("/link/{slug}")
def share_link(slug: str):
    return RedirectResponse(f"/a/{slug}")
