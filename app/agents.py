from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from slugify import slugify
from .db import get_db
from .models import Agent, Conversation, GoogleToken
from .security import get_current_user_id
from .rag import reindex_agent
from .scheduler import _creds_path_for_user

router = APIRouter(prefix="/agents", tags=["agents"])

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
