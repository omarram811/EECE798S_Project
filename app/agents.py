from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from slugify import slugify
from .db import get_db
from .models import Agent, Conversation, GoogleToken
from .security import get_current_user_id
from .rag import reindex_agent
from .scheduler import _creds_path_for_user
from fastapi.templating import Jinja2Templates
from .models import QueryLog  # Make sure this exists in models.py
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import json
import tempfile

router = APIRouter(prefix="/agents", tags=["agents"])
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

@router.post("/create")
def create_agent(request: Request,
                 name: str = Form(...),
                 drive_folder: str = Form(""),
                 persona: str = Form(""),
                 provider: str = Form("openai"),
                 model: str = Form("gpt-4o-mini"),
                 embed_model: str = Form("openai:text-embedding-3-small"),
                 db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Login required")
    slug = slugify(name)
    a = Agent(owner_id=uid, name=name, slug=slug, drive_folder_id=drive_folder, persona=persona,
              provider=provider, model=model, embed_model=embed_model)
    db.add(a); db.commit()
    # initial index if Drive connected and token present
    tok = db.query(GoogleToken).filter_by(user_id=uid).first()
    if tok and drive_folder:
        creds_path = _creds_path_for_user(uid)
        try:
            reindex_agent(a, creds_path)
        except Exception as e:
            print("Initial reindex failed:", e)
    return RedirectResponse(f"/a/{slug}", status_code=302)

@router.post("/{agent_id}/update")
def update_agent(request: Request, agent_id: int,
                 name: str = Form(...),
                 drive_folder: str = Form(""),
                 persona: str = Form(""),
                 provider: str = Form("openai"),
                 model: str = Form("gpt-4o-mini"),
                 embed_model: str = Form("openai:text-embedding-3-small"),
                 db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    a = db.get(Agent, agent_id)
    if not a or a.owner_id != uid:
        raise HTTPException(status_code=404)
    a.name, a.persona, a.drive_folder_id = name, persona, drive_folder
    a.provider, a.model, a.embed_model = provider, model, embed_model
    db.commit()
    return RedirectResponse(f"/a/{a.slug}", status_code=302)


@router.get("/{agent_id}/logs")
def view_logs(request: Request, agent_id: int, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Security: Only the professor who owns this agent
    if agent.owner_id != uid:
        raise HTTPException(status_code=403, detail="Not allowed")

    logs = (
        db.query(QueryLog)
        .filter_by(agent_id=agent_id)
        .order_by(QueryLog.timestamp.desc())
        .all()
    )

    return templates.TemplateResponse(
        "agent_logs.html",
        {
            "request": request,
            "agent": agent,
            "logs": logs
        }
    )




@router.get("/{agent_id}/logs/download")
def download_logs(agent_id: int, request: Request, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=404)

    logs = (
        db.query(QueryLog)
        .filter_by(agent_id=agent_id)
        .order_by(QueryLog.timestamp.asc())
        .all()
    )

    # Convert logs to list of dicts
    data = [
        {
            "timestamp": log.timestamp.isoformat(),
            "query": log.query,
            "response": log.response,
        }
        for log in logs
    ]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2)
        temp_path = tmp.name


    # Return file for download
    return FileResponse(
        tmp.name,
        media_type="application/json",
        filename=f"{agent.name}_logs.json"
    )