"""
Supabase Database Connection Module

Handles connection to Supabase PostgreSQL database using the Supabase REST API.
Credentials are read from config.py environment variables.
"""

import requests
from ..config import SUPABASE_URL, SUPABASE_SERVICE_KEY


class Database:
    """
    Supabase database connection manager.
    
    Uses the Supabase REST API directly via requests.
    Service key provides admin-level access from backend.
    """
    
    def __init__(self, url: str = SUPABASE_URL, key: str = SUPABASE_SERVICE_KEY):
        """
        Initialize Supabase connection.
        
        Args:
            url: Supabase project URL.
            key: Supabase service key (for server-side admin access).
        
        Raises:
            ValueError: If credentials are missing.
        """
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

        self.base_url = url.rstrip("/")
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()

        print(f"Supabase connection configured: {self.base_url}")

    def close(self):
        """
        Close the underlying HTTP session.
        """
        self.session.close()

    def close_all(self):
        """
        Alias for backward compatibility with shutdown cleanup.
        """
        self.close()

    def insert(self, table: str, data: dict) -> list:
        """
        Insert a record into a Supabase table and return the inserted row.
        """
        url = f"{self.base_url}/rest/v1/{table}"
        headers = self.headers.copy()
        headers["Prefer"] = "return=representation"

        response = self.session.post(url, json=data, headers=headers)
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Supabase insert failed ({response.status_code}): {response.text}")

        return response.json()

    def query(self, table: str, select: str = "*", params: dict = None) -> list:
        """
        Run a simple select query against a Supabase table.
        """
        url = f"{self.base_url}/rest/v1/{table}"
        query_params = {"select": select}
        if params:
            query_params.update(params)

        response = self.session.get(url, headers=self.headers, params=query_params)
        if response.status_code != 200:
            raise RuntimeError(f"Supabase query failed ({response.status_code}): {response.text}")

        return response.json()