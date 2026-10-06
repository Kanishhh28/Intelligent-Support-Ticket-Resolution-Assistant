import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def get_connection():
    """Create a PostgreSQL connection with pgvector support."""

    conn = psycopg.connect(settings.DATABASE_URL)

    register_vector(conn)

    return conn