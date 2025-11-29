# tests/test_progress.py
"""
Unit tests for app/progress.py

Tests cover:
- Progress tracking for embedding operations
- Thread-safe progress updates
- Progress percentage calculations
"""

import pytest
from datetime import datetime


class TestEmbeddingProgress:
    """Tests for EmbeddingProgress dataclass."""

    def test_progress_percentage_zero_files(self):
        """Test progress percentage when total_files is zero."""
        from app.progress import EmbeddingProgress
        
        progress = EmbeddingProgress(
            agent_id=1,
            total_files=0,
            processed_files=0,
            current_file="",
            status="processing"
        )
        
        assert progress.progress_percentage == 0.0

    def test_progress_percentage_partial(self):
        """Test progress percentage with partial completion."""
        from app.progress import EmbeddingProgress
        
        progress = EmbeddingProgress(
            agent_id=1,
            total_files=10,
            processed_files=5,
            current_file="file5.pdf",
            status="processing"
        )
        
        assert progress.progress_percentage == 50.0

    def test_progress_percentage_complete(self):
        """Test progress percentage at 100%."""
        from app.progress import EmbeddingProgress
        
        progress = EmbeddingProgress(
            agent_id=1,
            total_files=10,
            processed_files=10,
            current_file="",
            status="completed"
        )
        
        assert progress.progress_percentage == 100.0

    def test_progress_force_reindex_default(self):
        """Test force_reindex defaults to False."""
        from app.progress import EmbeddingProgress
        
        progress = EmbeddingProgress(
            agent_id=1,
            total_files=5,
            processed_files=0,
            current_file="",
            status="processing"
        )
        
        assert progress.force_reindex is False


class TestProgressTracking:
    """Tests for progress tracking functions."""

    def test_start_progress(self):
        """Test starting progress tracking."""
        from app.progress import start_progress, get_progress, clear_progress
        
        try:
            start_progress(agent_id=999, total_files=10, force_reindex=True)
            
            progress = get_progress(999)
            
            assert progress is not None
            assert progress.agent_id == 999
            assert progress.total_files == 10
            assert progress.processed_files == 0
            assert progress.status == "processing"
            assert progress.force_reindex is True
            assert progress.started_at is not None
        finally:
            clear_progress(999)

    def test_update_progress(self):
        """Test updating progress."""
        from app.progress import start_progress, update_progress, get_progress, clear_progress
        
        try:
            start_progress(agent_id=998, total_files=10)
            
            update_progress(agent_id=998, processed_files=5, current_file="file5.pdf")
            
            progress = get_progress(998)
            
            assert progress.processed_files == 5
            assert progress.current_file == "file5.pdf"
        finally:
            clear_progress(998)

    def test_complete_progress_success(self):
        """Test completing progress successfully."""
        from app.progress import start_progress, complete_progress, get_progress, clear_progress
        
        try:
            start_progress(agent_id=997, total_files=10)
            
            complete_progress(agent_id=997)
            
            progress = get_progress(997)
            
            assert progress.status == "completed"
            assert progress.error_message is None
            assert progress.completed_at is not None
        finally:
            clear_progress(997)

    def test_complete_progress_with_error(self):
        """Test completing progress with error."""
        from app.progress import start_progress, complete_progress, get_progress, clear_progress
        
        try:
            start_progress(agent_id=996, total_files=10)
            
            complete_progress(agent_id=996, error="Connection failed")
            
            progress = get_progress(996)
            
            assert progress.status == "error"
            assert progress.error_message == "Connection failed"
        finally:
            clear_progress(996)

    def test_get_progress_nonexistent(self):
        """Test getting progress for non-existent agent."""
        from app.progress import get_progress
        
        progress = get_progress(12345)
        
        assert progress is None

    def test_clear_progress(self):
        """Test clearing progress."""
        from app.progress import start_progress, clear_progress, get_progress
        
        start_progress(agent_id=995, total_files=5)
        clear_progress(995)
        
        progress = get_progress(995)
        
        assert progress is None

    def test_is_force_reindex_true(self):
        """Test is_force_reindex returns True when set."""
        from app.progress import start_progress, is_force_reindex, clear_progress
        
        try:
            start_progress(agent_id=994, total_files=5, force_reindex=True)
            
            assert is_force_reindex(994) is True
        finally:
            clear_progress(994)

    def test_is_force_reindex_false(self):
        """Test is_force_reindex returns False when not set."""
        from app.progress import start_progress, is_force_reindex, clear_progress
        
        try:
            start_progress(agent_id=993, total_files=5, force_reindex=False)
            
            assert is_force_reindex(993) is False
        finally:
            clear_progress(993)

    def test_is_force_reindex_nonexistent(self):
        """Test is_force_reindex returns False for non-existent agent."""
        from app.progress import is_force_reindex
        
        assert is_force_reindex(99999) is False
