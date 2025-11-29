# app/server_state.py
import uuid

# Generate a unique server instance ID on startup
# This is used to detect server restarts and force re-authentication
SERVER_INSTANCE_ID = str(uuid.uuid4())
