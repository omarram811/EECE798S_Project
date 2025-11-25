from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
from contextlib import contextmanager
from .db import SessionLocal
from .models import Agent, GoogleToken
from .rag import reindex_agent
from .settings import get_settings
import os, json

@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()

def _creds_path_for_user(user_id: int) -> str:
    # store token per user on disk to reuse with loaders
    path = f"data/google_tokens/user_{user_id}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with session_scope() as db:
        tok = db.query(GoogleToken).filter_by(user_id=user_id).first()
        if tok:
            with open(path, "w", encoding="utf-8") as f:
                f.write(tok.token_json)
    return path

def daily_refresh_job():
    print("[refresh] Job started")
    with session_scope() as db:
        agents = db.query(Agent).all()
        print(f"[refresh] Found {len(agents)} agents")
        for a in agents:
            if not a.drive_folder_id:
                print(f"[refresh] Skipping agent {a.id}, no drive_folder_id")
                continue
            creds_path = _creds_path_for_user(a.owner_id)
            try:
                # Pass track_progress=False for background scheduler updates
                reindex_agent(a, creds_path, track_progress=False)
                print(f"[refresh] refresh job finished for agent {a.id}")
            except Exception as e:
                print(f"[refresh] agent {a.id} failed: {e}")
    print("[refresh] Job finished")
    

scheduler = BackgroundScheduler()

# def start_scheduler():
#     cron = get_settings().SCHED_CRON
#     parts = cron.split()
#     trig = CronTrigger.from_crontab(cron)
#     scheduler.add_job(daily_refresh_job, trig, id="daily_refresh", replace_existing=True)
#     scheduler.start()


from apscheduler.triggers.interval import IntervalTrigger

def start_scheduler():
    trigger = IntervalTrigger(minutes=2)  # every 2 minutes from now
    scheduler.add_job(daily_refresh_job, trigger, id="daily_refresh", replace_existing=True)
    print("[scheduler] Jobs added:", scheduler.get_jobs())
    scheduler.start()

