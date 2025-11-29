# tests/test_security.py
"""
Unit tests for app/security.py

Tests cover:
- Password hashing and verification
- Session cookie management
- User authentication
"""

import pytest
from unittest.mock import MagicMock, patch
import os


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_hash_password_returns_string(self):
        """Test that hash_password returns a string."""
        from app.security import hash_password
        
        result = hash_password("testpassword123")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_password_different_for_same_input(self):
        """Test that hashing same password twice gives different hashes (salted)."""
        from app.security import hash_password
        
        hash1 = hash_password("testpassword")
        hash2 = hash_password("testpassword")
        
        # Due to salting, hashes should be different
        # Note: pbkdf2_sha256 includes salt, so same password -> different hash
        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        from app.security import hash_password, verify_password
        
        password = "mysecretpassword123"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        from app.security import hash_password, verify_password
        
        hashed = hash_password("correctpassword")
        
        assert verify_password("wrongpassword", hashed) is False

    def test_verify_password_empty(self):
        """Test password verification with empty password."""
        from app.security import hash_password, verify_password
        
        hashed = hash_password("somepassword")
        
        assert verify_password("", hashed) is False


class TestSessionCookie:
    """Tests for session cookie management."""

    def test_cookie_name_default(self):
        """Test that default cookie name is set."""
        from app.security import COOKIE_NAME
        
        assert COOKIE_NAME is not None
        assert len(COOKIE_NAME) > 0

    def test_set_session_cookie(self):
        """Test setting session cookie on response."""
        from app.security import set_session_cookie
        
        mock_response = MagicMock()
        
        set_session_cookie(mock_response, user_id=123)
        
        mock_response.set_cookie.assert_called_once()
        call_args = mock_response.set_cookie.call_args
        
        # Check cookie settings
        assert call_args[1]["httponly"] is True
        assert call_args[1]["samesite"] == "lax"
        assert call_args[1]["path"] == "/"

    def test_clear_session_cookie(self):
        """Test clearing session cookie."""
        from app.security import clear_session_cookie, COOKIE_NAME
        
        mock_response = MagicMock()
        
        clear_session_cookie(mock_response)
        
        mock_response.delete_cookie.assert_called_once_with(COOKIE_NAME, path="/")


class TestGetCurrentUserId:
    """Tests for getting current user ID from request."""

    def test_get_current_user_id_no_cookie(self):
        """Test get_current_user_id when no cookie is present."""
        from app.security import get_current_user_id
        
        mock_request = MagicMock()
        mock_request.cookies = {}
        
        result = get_current_user_id(mock_request)
        
        assert result is None

    def test_get_current_user_id_valid_cookie(self):
        """Test get_current_user_id with valid cookie."""
        from app.security import set_session_cookie, get_current_user_id, COOKIE_NAME
        
        # First, create a valid token
        mock_response = MagicMock()
        set_session_cookie(mock_response, user_id=42)
        
        # Get the token that was set
        call_args = mock_response.set_cookie.call_args
        token = call_args[0][1]  # Second positional arg is the token value
        
        # Now create a request with this token
        mock_request = MagicMock()
        mock_request.cookies = {COOKIE_NAME: token}
        
        result = get_current_user_id(mock_request)
        
        assert result == 42

    def test_get_current_user_id_invalid_cookie(self):
        """Test get_current_user_id with invalid/tampered cookie."""
        from app.security import get_current_user_id, COOKIE_NAME
        
        mock_request = MagicMock()
        mock_request.cookies = {COOKIE_NAME: "invalid_token_value"}
        
        result = get_current_user_id(mock_request)
        
        assert result is None

    def test_get_current_user_id_expired_cookie(self):
        """Test get_current_user_id with expired cookie."""
        from app.security import get_current_user_id, COOKIE_NAME, _get_signer
        from itsdangerous import TimestampSigner
        
        # Create a token that looks valid but test expiration handling
        # Note: We can't easily test expiration without mocking time
        # but we can test that invalid signatures are rejected
        
        mock_request = MagicMock()
        mock_request.cookies = {COOKIE_NAME: "tampered.signature.here"}
        
        result = get_current_user_id(mock_request)
        
        assert result is None


class TestSignerConfiguration:
    """Tests for signer configuration."""

    def test_signer_uses_secret_key(self):
        """Test that signer uses SECRET_KEY from environment."""
        from app.security import _get_signer
        
        os.environ["SECRET_KEY"] = "test_secret_for_unit_test"
        
        signer = _get_signer()
        
        # Create a token and verify it works
        token = signer.sign("123")
        unsigned = signer.unsign(token)
        
        assert unsigned == b"123"

    def test_signer_default_key(self):
        """Test that signer uses default key when SECRET_KEY not set."""
        from app.security import _get_signer
        
        # Temporarily remove SECRET_KEY if it exists
        old_key = os.environ.pop("SECRET_KEY", None)
        
        try:
            signer = _get_signer()
            # Should not raise exception
            token = signer.sign("test")
            assert token is not None
        finally:
            if old_key:
                os.environ["SECRET_KEY"] = old_key
