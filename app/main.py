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
templates = Jinja2Templates(directory="templates")


from fastapi import FastAPI
from contextlib import asynccontextmanager
from .scheduler import start_scheduler
from .server_state import SERVER_INSTANCE_ID


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("[lifespan] Starting scheduler...")
    print(f"[lifespan] Server instance ID: {SERVER_INSTANCE_ID}")
    start_scheduler()
    yield
    # SHUTDOWN
    print("[lifespan] Shutting down scheduler...")
    # optional: scheduler.shutdown() if you want graceful shutdown

# Create a single app with lifespan
app = FastAPI(title="Course TA Agent Studio", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Middleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY","dev"))

# Routers
app.include_router(auth_router)
app.include_router(agents_router)
app.include_router(chat_router)
app.include_router(google_router)


@app.get("/", response_class=HTMLResponse)
def home(request: Request, error: str = None):
    uid = get_current_user_id(request)

    # Check if user has the current server instance in their session
    # If not, clear their authentication and force re-login
    if uid:
        session_instance = request.session.get("server_instance_id")
        if session_instance != SERVER_INSTANCE_ID:
            # Server was restarted, force re-login
            request.session.clear()
            return templates.TemplateResponse("home.html", {
                "request": request,
                "user_id": None,
                "error": error
            })
        return RedirectResponse("/dashboard")

    # Else show login/register page with error parameter
    return templates.TemplateResponse("home.html", {
        "request": request,
        "user_id": None,
        "error": error
    })


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        return RedirectResponse("/")
    
    # Verify server instance matches (force re-login on server restart)
    session_instance = request.session.get("server_instance_id")
    if session_instance != SERVER_INSTANCE_ID:
        request.session.clear()
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
    
    # Get full URL for sharing
    base_url = str(request.base_url).rstrip('/')
    share_url = f"{base_url}/public/{slug}"
    
    return templates.TemplateResponse("chat.html", {"request": request, "agent": agent, "messages": messages, "share_url": share_url})

# Public chat page (no authentication required)
@app.get("/public/{slug}", response_class=HTMLResponse)
def public_chat_page(slug: str, request: Request, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter_by(slug=slug).first()
    if not agent:
        raise HTTPException(status_code=404)
    
    # For public users, we'll use a session-based approach
    session_id = request.session.get("public_session_id")
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        request.session["public_session_id"] = session_id
    
    # Store public conversations with session_id in user_id field as negative hash
    public_user_id = -abs(hash(session_id)) % (10 ** 8)
    
    conv = db.query(Conversation).filter_by(agent_id=agent.id, user_id=public_user_id).order_by(Conversation.id.desc()).first()
    messages = []
    if conv:
        messages = db.query(Message).filter_by(conversation_id=conv.id).order_by(Message.id.asc()).all()
    
    return templates.TemplateResponse("public_chat.html", {"request": request, "agent": agent, "messages": messages})
