from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from .db import get_db
from .models import Agent, Conversation, GoogleToken, AgentFile
from .security import get_current_user_id
from .rag import reindex_agent
from .scheduler import _creds_path_for_user
from fastapi.templating import Jinja2Templates
from .models import QueryLog  # Make sure this exists in models.py
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import json
import tempfile
import uuid
from .progress import get_progress
import threading

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
                 api_key: str = Form(...),
                 db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Login required")
    
    # Check if Google Drive is connected - required to create agents
    tok = db.query(GoogleToken).filter_by(user_id=uid).first()
    if not tok:
        raise HTTPException(
            status_code=400, 
            detail="Google Drive not connected. Please connect your Google Drive before creating an agent."
        )
    
    slug = str(uuid.uuid4())
    a = Agent(owner_id=uid, name=name, slug=slug, drive_folder_id=drive_folder, persona=persona,
              provider=provider, model=model, embed_model=embed_model, api_key=api_key)
    db.add(a); db.commit()
    
    # Start indexing in background thread if Drive folder provided
    if drive_folder:
        agent_id = a.id  # Store agent ID for background thread
        creds_path = _creds_path_for_user(uid)
        # Run in background thread with its own DB session
        def background_index():
            from .db import SessionLocal
            db_bg = SessionLocal()
            try:
                # Fetch agent in background thread's session
                agent_bg = db_bg.get(Agent, agent_id)
                if agent_bg:
                    # Enable progress tracking for initial agent creation
                    reindex_agent(agent_bg, creds_path, track_progress=True)
                    print(f"[create_agent] Background indexing completed for agent {agent_id}")
                else:
                    print(f"[create_agent] Agent {agent_id} not found in background thread")
            except Exception as e:
                print(f"[create_agent] Background indexing failed for agent {agent_id}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                db_bg.close()
        
        thread = threading.Thread(target=background_index, daemon=True)
        thread.start()
        
        # Redirect to progress page to show indexing progress
        return RedirectResponse(f"/agents/{a.id}/progress-page", status_code=302)
    
    # If no drive folder, go directly to agent
    return RedirectResponse(f"/a/{slug}", status_code=302)


@router.post("/{agent_id}/delete")
def delete_agent(request: Request, agent_id: int, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=404, detail="Agent not found")

    # 1. Delete conversations (cascade deletes messages)
    for conv in agent.conversations:
        db.delete(conv)

    # 2. Delete query logs
    for log in agent.query_logs:
        db.delete(log)

    # 3. AgentFiles will delete automatically via cascade now

    # 4. Delete vector embeddings
    try:
        from .rag import ensure_collection, get_vector_client
        client = get_vector_client()
        col = ensure_collection(agent, client)
        col.delete(where={"agent_id": agent_id})
    except Exception as e:
        print("[delete_agent] Warning:", e)

    # 5. Delete the agent itself
    db.delete(agent)
    db.commit()

    db.query(Agent).all()
    db.query(Conversation).all()
    db.query(QueryLog).all()
    db.query(AgentFile).all()

    return RedirectResponse("/dashboard", status_code=302)




@router.post("/{agent_id}/update")
def update_agent(request: Request, agent_id: int,
                 name: str = Form(...),
                 drive_folder: str = Form(""),
                 persona: str = Form(""),
                 provider: str = Form("openai"),
                 model: str = Form("gpt-4o-mini"),
                 embed_model: str = Form("openai:text-embedding-3-small"),
                 api_key: str = Form(""),
                 db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    a = db.get(Agent, agent_id)
    if not a or a.owner_id != uid:
        raise HTTPException(status_code=404)
    
    # Check if embedding model or provider changed
    embedding_changed = (a.provider != provider or a.embed_model != embed_model)
    
    a.name, a.persona, a.drive_folder_id = name, persona, drive_folder
    a.provider, a.model, a.embed_model = provider, model, embed_model
    if api_key:  # Only update API key if provided
        a.api_key = api_key
    db.commit()
    
    # If embedding model changed, trigger re-indexing to rebuild vector store
    if embedding_changed and drive_folder:
        tok = db.query(GoogleToken).filter_by(user_id=uid).first()
        if tok:
            stored_agent_id = a.id  # Store agent ID for background thread
            creds_path = _creds_path_for_user(uid)
            
            # Run re-indexing in background thread with progress tracking
            def background_reindex():
                from .db import SessionLocal
                db_bg = SessionLocal()
                try:
                    # Fetch agent in background thread's session
                    agent_bg = db_bg.get(Agent, stored_agent_id)
                    if agent_bg:
                        print(f"[update_agent] Embedding model changed for agent {stored_agent_id}. Re-indexing with progress tracking...")
                        reindex_agent(agent_bg, creds_path, track_progress=True)
                        print(f"[update_agent] Background re-indexing completed for agent {stored_agent_id}")
                    else:
                        print(f"[update_agent] Agent {stored_agent_id} not found in background thread")
                except Exception as e:
                    print(f"[update_agent] Background re-indexing failed for agent {stored_agent_id}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    db_bg.close()
            
            thread = threading.Thread(target=background_reindex, daemon=True)
            thread.start()
            
            # Redirect to progress page to show re-indexing progress
            return RedirectResponse(f"/agents/{a.id}/progress-page", status_code=302)
    
    # No embedding change, go back to chat
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


@router.get("/{agent_id}/progress")
def get_embedding_progress(agent_id: int, request: Request, db: Session = Depends(get_db)):
    """Get the current embedding progress for an agent"""
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=404)

    progress = get_progress(agent_id)
    if not progress:
        return JSONResponse({
            "status": "idle",
            "total_files": 0,
            "processed_files": 0,
            "current_file": "",
            "progress_percentage": 0
        })

    return JSONResponse({
        "status": progress.status,
        "total_files": progress.total_files,
        "processed_files": progress.processed_files,
        "current_file": progress.current_file,
        "progress_percentage": progress.progress_percentage,
        "error_message": progress.error_message
    })


@router.get("/{agent_id}/progress-page")
def show_progress_page(agent_id: int, request: Request, db: Session = Depends(get_db)):
    """Show the progress page for initial agent setup"""
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=404)

    return templates.TemplateResponse("agent_progress.html", {
        "request": request,
        "agent": agent
    })