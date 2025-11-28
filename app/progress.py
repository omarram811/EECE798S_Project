# app/progress.py
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import threading

@dataclass
class EmbeddingProgress:
    agent_id: int
    total_files: int
    processed_files: int
    current_file: str
    status: str  # 'processing', 'completed', 'error'
    error_message: Optional[str] = None
    started_at: datetime = None
    completed_at: Optional[datetime] = None
    force_reindex: bool = False  # If True, forces full re-embedding regardless of cache

    @property
    def progress_percentage(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100

# Global progress tracker
_progress_lock = threading.Lock()
_progress_tracker: Dict[int, EmbeddingProgress] = {}

def start_progress(agent_id: int, total_files: int, force_reindex: bool = False):
    """Initialize progress tracking for an agent"""
    with _progress_lock:
        _progress_tracker[agent_id] = EmbeddingProgress(
            agent_id=agent_id,
            total_files=total_files,
            processed_files=0,
            current_file="",
            status="processing",
            started_at=datetime.utcnow(),
            force_reindex=force_reindex
        )


def is_force_reindex(agent_id: int) -> bool:
    """Check if force reindex is enabled for an agent"""
    with _progress_lock:
        progress = _progress_tracker.get(agent_id)
        return progress.force_reindex if progress else False

def update_progress(agent_id: int, processed_files: int, current_file: str = ""):
    """Update progress for an agent"""
    with _progress_lock:
        if agent_id in _progress_tracker:
            _progress_tracker[agent_id].processed_files = processed_files
            _progress_tracker[agent_id].current_file = current_file

def complete_progress(agent_id: int, error: Optional[str] = None):
    """Mark progress as completed or failed"""
    with _progress_lock:
        if agent_id in _progress_tracker:
            _progress_tracker[agent_id].status = "error" if error else "completed"
            _progress_tracker[agent_id].error_message = error
            _progress_tracker[agent_id].completed_at = datetime.utcnow()

def get_progress(agent_id: int) -> Optional[EmbeddingProgress]:
    """Get current progress for an agent"""
    with _progress_lock:
        return _progress_tracker.get(agent_id)

def clear_progress(agent_id: int):
    """Clear progress tracking for an agent"""
    with _progress_lock:
        if agent_id in _progress_tracker:
            del _progress_tracker[agent_id]
