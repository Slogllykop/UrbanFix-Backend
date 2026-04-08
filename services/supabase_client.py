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


def update_validation_status(
    row_id: str, 
    status: str, 
    ai_verified: bool, 
    category: str = None
) -> dict:
    """
    Update the 'status', 'ai_verified', and other relevant columns for an issue.

    Args:
        row_id:  Primary-key UUID of the issue to update.
        status:  One of "verified", "rejected", "pending", or "addressed".
        ai_verified: Boolean indicating if it was AI verified.
        category: Optional category string (e.g., "pothole", "garbage").

    Returns:
        The Supabase response data.

    Raises:
        Exception: If the update query fails.
    """
    supabase = get_supabase_client()
    
    # Prepare update payload based on the issues table schema
    update_data = {
        "status": status,
        "ai_verified": ai_verified,
    }
    
    if category:
        update_data["category"] = category
        
    # Automatically set/increment priority if verified by AI
    if status == "verified" and ai_verified:
        update_data["priority_score"] = 10  # Base priority for AI-verified issues

    response = (
        supabase.table(config.TABLE_NAME)
        .update(update_data)
        .eq("id", row_id)
        .execute()
    )
    return response.data
