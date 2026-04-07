"""
Supabase client – handles image downloads and database updates.
"""

import httpx
from supabase import create_client, Client

import config


def get_supabase_client() -> Client:
    """Create and return a Supabase client instance."""
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


async def fetch_image_bytes(image_url: str) -> bytes:
    """
    Download an image from the given Supabase Storage URL.

    The frontend provides a full public URL to the image stored in a
    Supabase Storage bucket. We simply fetch it over HTTP.

    Args:
        image_url: Full public URL of the image in Supabase Storage.

    Returns:
        Raw image bytes.

    Raises:
        httpx.HTTPStatusError: If the download fails.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(image_url)
        response.raise_for_status()
        return response.content


def update_validation_status(row_id: str, status: str) -> dict:
    """
    Update the 'validate' column for a specific row in the database.

    Args:
        row_id:  Primary-key ID of the row to update.
        status:  "valid" or "invalid".

    Returns:
        The Supabase response data.

    Raises:
        Exception: If the update query fails.
    """
    supabase = get_supabase_client()
    response = (
        supabase.table(config.TABLE_NAME)
        .update({"validate": status})
        .eq("id", row_id)
        .execute()
    )
    return response.data
