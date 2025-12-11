"""
Comprehensive tool definitions following LangGraph best practices.
All tools are scoped to Customer ID 58 (Manoj Pareek) for testing.
"""

import os
import requests
from langchain_core.tools import tool
from database import get_database
from googleapiclient.discovery import build

# Initialize database
db = get_database()

# Default test customer
DEFAULT_CUSTOMER_ID = 58  # Manoj Pareek - has lots of data


# =======================
# CUSTOMER ACCOUNT TOOLS
# =======================

@tool
def get_customer_account():
    """
    Get the current customer's account details including name, email, address, phone.
    """
    result = db.run(
        f"""
        SELECT 
            CustomerId, FirstName, LastName, Email, 
            Company, Address, City, State, Country, 
            PostalCode, Phone, Fax
        FROM Customer 
        WHERE CustomerId = {DEFAULT_CUSTOMER_ID};
        """,
        include_columns=True
    )
    return result


@tool
def get_invoice_history():
    """
    Get the customer's complete invoice/order history with dates and totals.
    Shows all purchases made by the customer.
    """
    result = db.run(
        f"""
        SELECT 
            i.InvoiceId,
            i.InvoiceDate,
            i.Total,
            i.BillingAddress,
            i.BillingCity,
            i.BillingCountry,
            COUNT(il.InvoiceLineId) as TotalItems
        FROM Invoice i
        LEFT JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
        WHERE i.CustomerId = {DEFAULT_CUSTOMER_ID}
        GROUP BY i.InvoiceId
        ORDER BY i.InvoiceDate DESC;
        """,
        include_columns=True
    )
    return result


@tool
def get_purchased_tracks():
    """
    Get all tracks/songs the customer has purchased with details like artist, album, price.
    """
    result = db.run(
        f"""
        SELECT DISTINCT
            t.TrackId,
            t.Name as TrackName,
            a.Title as AlbumName,
            ar.Name as ArtistName,
            g.Name as Genre,
            t.Composer,
            t.Milliseconds,
            il.UnitPrice as PricePaid,
            i.InvoiceDate as PurchaseDate
        FROM Invoice i
        JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
        JOIN Track t ON il.TrackId = t.TrackId
        LEFT JOIN Album a ON t.AlbumId = a.AlbumId
        LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
        LEFT JOIN Genre g ON t.GenreId = g.GenreId
        WHERE i.CustomerId = {DEFAULT_CUSTOMER_ID}
        ORDER BY i.InvoiceDate DESC;
        """,
        include_columns=True
    )
    return result


@tool
def get_spending_summary():
    """
    Get a summary of the customer's total spending, number of orders, and average order value.
    """
    result = db.run(
        f"""
        SELECT 
            COUNT(DISTINCT i.InvoiceId) as TotalOrders,
            SUM(i.Total) as TotalSpent,
            AVG(i.Total) as AverageOrderValue,
            COUNT(DISTINCT il.TrackId) as UniqueTracks,
            MIN(i.InvoiceDate) as FirstPurchase,
            MAX(i.InvoiceDate) as LastPurchase
        FROM Invoice i
        JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
        WHERE i.CustomerId = {DEFAULT_CUSTOMER_ID};
        """,
        include_columns=True
    )
    return result


@tool
def get_invoice_details(invoice_id: int):
    """
    Get detailed line items for a specific invoice/order.
    
    Args:
        invoice_id: The ID of the invoice to look up
    """
    result = db.run(
        f"""
        SELECT 
            i.InvoiceId,
            i.InvoiceDate,
            i.Total as InvoiceTotal,
            t.Name as TrackName,
            a.Title as AlbumName,
            ar.Name as ArtistName,
            il.UnitPrice,
            il.Quantity,
            (il.UnitPrice * il.Quantity) as LineTotal
        FROM Invoice i
        JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
        JOIN Track t ON il.TrackId = t.TrackId
        LEFT JOIN Album a ON t.AlbumId = a.AlbumId
        LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
        WHERE i.InvoiceId = {invoice_id}
        AND i.CustomerId = {DEFAULT_CUSTOMER_ID};
        """,
        include_columns=True
    )
    return result


# =======================
# MUSIC CATALOG TOOLS
# =======================

@tool
def search_tracks(search_term: str):
    """
    Search for tracks/songs by name. Supports partial matching.
    
    Args:
        search_term: The track/song name to search for
    """
    result = db.run(
        f"""
        SELECT 
            t.TrackId,
            t.Name as TrackName,
            a.Title as AlbumName,
            ar.Name as ArtistName,
            g.Name as Genre,
            t.Composer,
            ROUND(t.Milliseconds / 60000.0, 2) as DurationMinutes,
            t.UnitPrice
        FROM Track t
        LEFT JOIN Album a ON t.AlbumId = a.AlbumId
        LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
        LEFT JOIN Genre g ON t.GenreId = g.GenreId
        WHERE t.Name LIKE '%{search_term}%'
        LIMIT 20;
        """,
        include_columns=True
    )
    return result


@tool
def search_artists(artist_name: str):
    """
    Search for artists by name and get their album count. Supports partial matching.
    
    Args:
        artist_name: The artist name to search for
    """
    result = db.run(
        f"""
        SELECT 
            ar.ArtistId,
            ar.Name as ArtistName,
            COUNT(DISTINCT a.AlbumId) as AlbumCount,
            COUNT(DISTINCT t.TrackId) as TrackCount
        FROM Artist ar
        LEFT JOIN Album a ON ar.ArtistId = a.ArtistId
        LEFT JOIN Track t ON a.AlbumId = t.AlbumId
        WHERE ar.Name LIKE '%{artist_name}%'
        GROUP BY ar.ArtistId, ar.Name
        LIMIT 15;
        """,
        include_columns=True
    )
    return result


@tool
def get_artist_albums(artist_name: str):
    """
    Get all albums by an artist with track counts and pricing info.
    
    Args:
        artist_name: The artist name to look up
    """
    result = db.run(
        f"""
        SELECT 
            a.AlbumId,
            a.Title as AlbumName,
            ar.Name as ArtistName,
            COUNT(t.TrackId) as TrackCount,
            MIN(t.UnitPrice) as MinPrice,
            MAX(t.UnitPrice) as MaxPrice
        FROM Album a
        JOIN Artist ar ON a.ArtistId = ar.ArtistId
        LEFT JOIN Track t ON a.AlbumId = t.AlbumId
        WHERE ar.Name LIKE '%{artist_name}%'
        GROUP BY a.AlbumId, a.Title, ar.Name
        ORDER BY a.Title;
        """,
        include_columns=True
    )
    return result


@tool
def get_album_tracks(album_name: str):
    """
    Get all tracks from a specific album with full details.
    
    Args:
        album_name: The album name to look up
    """
    result = db.run(
        f"""
        SELECT 
            t.TrackId,
            t.Name as TrackName,
            a.Title as AlbumName,
            ar.Name as ArtistName,
            g.Name as Genre,
            t.Composer,
            ROUND(t.Milliseconds / 60000.0, 2) as DurationMinutes,
            t.UnitPrice
        FROM Track t
        JOIN Album a ON t.AlbumId = a.AlbumId
        JOIN Artist ar ON a.ArtistId = ar.ArtistId
        LEFT JOIN Genre g ON t.GenreId = g.GenreId
        WHERE a.Title LIKE '%{album_name}%'
        ORDER BY t.TrackId;
        """,
        include_columns=True
    )
    return result


@tool
def get_genres():
    """
    Get all available music genres with track and album counts.
    """
    result = db.run(
        """
        SELECT 
            g.GenreId,
            g.Name as GenreName,
            COUNT(DISTINCT t.TrackId) as TrackCount,
            COUNT(DISTINCT a.AlbumId) as AlbumCount
        FROM Genre g
        LEFT JOIN Track t ON g.GenreId = t.GenreId
        LEFT JOIN Album a ON t.AlbumId = a.AlbumId
        GROUP BY g.GenreId, g.Name
        ORDER BY TrackCount DESC;
        """,
        include_columns=True
    )
    return result


@tool
def get_tracks_by_genre(genre_name: str):
    """
    Get tracks filtered by genre with artist and pricing info.
    
    Args:
        genre_name: The genre name to filter by
    """
    result = db.run(
        f"""
        SELECT 
            t.TrackId,
            t.Name as TrackName,
            a.Title as AlbumName,
            ar.Name as ArtistName,
            g.Name as Genre,
            t.UnitPrice,
            ROUND(t.Milliseconds / 60000.0, 2) as DurationMinutes
        FROM Track t
        JOIN Genre g ON t.GenreId = g.GenreId
        LEFT JOIN Album a ON t.AlbumId = a.AlbumId
        LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
        WHERE g.Name LIKE '%{genre_name}%'
        ORDER BY ar.Name, a.Title, t.Name
        LIMIT 50;
        """,
        include_columns=True
    )
    return result


@tool
def get_popular_tracks():
    """
    Get the most popular tracks based on purchase count across all customers.
    """
    result = db.run(
        """
        SELECT 
            t.TrackId,
            t.Name as TrackName,
            a.Title as AlbumName,
            ar.Name as ArtistName,
            g.Name as Genre,
            t.UnitPrice,
            COUNT(il.InvoiceLineId) as PurchaseCount
        FROM Track t
        LEFT JOIN Album a ON t.AlbumId = a.AlbumId
        LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
        LEFT JOIN Genre g ON t.GenreId = g.GenreId
        LEFT JOIN InvoiceLine il ON t.TrackId = il.TrackId
        GROUP BY t.TrackId
        ORDER BY PurchaseCount DESC
        LIMIT 25;
        """,
        include_columns=True
    )
    return result


@tool
def get_track_price(track_name: str):
    """
    Look up the price of a specific track.
    
    Args:
        track_name: The track name to look up pricing for
    """
    result = db.run(
        f"""
        SELECT 
            t.Name as TrackName,
            a.Title as AlbumName,
            ar.Name as ArtistName,
            t.UnitPrice
        FROM Track t
        LEFT JOIN Album a ON t.AlbumId = a.AlbumId
        LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
        WHERE t.Name LIKE '%{track_name}%'
        LIMIT 10;
        """,
        include_columns=True
    )
    return result


# =======================
# LYRICS SEARCH TOOLS
# =======================

@tool
def search_song_by_lyrics(lyrics_snippet: str) -> str:
    """
    Search for songs using a snippet of lyrics. Use this when a customer provides lyrics they remember.
    Returns the top matching songs with artist and title information.
    
    Args:
        lyrics_snippet: A snippet of lyrics the customer remembers
        
    Returns:
        Top matching songs with title and artist
    """
    try:
        genius_token = os.getenv("GENIUS_ACCESS_TOKEN")
        if not genius_token:
            return "Error: Genius API token not configured. Please contact support."
        
        url = "https://api.genius.com/search"
        params = {
            "access_token": genius_token,
            "q": lyrics_snippet
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract songs from response
        hits = data.get('response', {}).get('hits', [])
        
        if not hits:
            return "No songs found matching those lyrics. Please try a different snippet."
        
        results = []
        for idx, hit in enumerate(hits[:5], 1):  # Top 5 results
            if hit.get('type') == 'song':
                result = hit.get('result', {})
                artist_info = result.get('primary_artist', {})
                
                results.append({
                    'rank': idx,
                    'title': result.get('title', 'Unknown').strip(),
                    'artist': artist_info.get('name', 'Unknown').strip()
                })
        
        if not results:
            return "No songs found matching those lyrics."
        
        # Format the response
        response_text = "Found these songs matching your lyrics:\n\n"
        for song in results:
            response_text += f"{song['rank']}. \"{song['title']}\" by {song['artist']}\n"
        
        return response_text
        
    except Exception as e:
        return f"Error searching for lyrics: {str(e)}"


@tool
def check_song_in_catalogue(song_title: str, artist_name: str) -> str:
    """
    Check if a specific song (by title and artist) exists in our music store catalogue.
    Use this after finding a song from lyrics to see if we have it available.
    
    Args:
        song_title: The title of the song
        artist_name: The name of the artist
        
    Returns:
        Whether the song is in our catalogue with details
    """
    try:
        # Search for the song in our database
        result = db.run(
            f"""
            SELECT Track.Name as SongName, Artist.Name as ArtistName, Album.Title as AlbumName,
                   Track.UnitPrice, Track.TrackId
            FROM Track
            JOIN Album ON Track.AlbumId = Album.AlbumId
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            WHERE Track.Name LIKE '%{song_title}%' AND Artist.Name LIKE '%{artist_name}%';
            """,
            include_columns=True
        )
        
        if result and "SongName" in result and len(result) > 2:
            return f"✅ YES! This song IS in our catalogue:\n\n{result}"
        else:
            return f"❌ Sorry, '{song_title}' by {artist_name} is NOT currently in our catalogue."
            
    except Exception as e:
        return f"Error checking catalogue: {str(e)}"


@tool
def search_youtube_video(song_title: str, artist_name: str) -> str:
    """
    Search for a song's official video or performance on YouTube.
    Returns the best matching video with video ID and details for embedding.
    
    Args:
        song_title: The title of the song
        artist_name: The name of the artist
        
    Returns:
        Video information including video_id for embedding
    """
    try:
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if not youtube_api_key:
            return "Error: YouTube API key not configured. Please contact support."
        
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        
        # Search for the video
        search_query = f"{song_title} {artist_name} official"
        search_response = youtube.search().list(
            q=search_query,
            part='id,snippet',
            maxResults=1,
            type='video',
            videoEmbeddable='true'
        ).execute()
        
        if not search_response.get('items'):
            return f"No YouTube videos found for '{song_title}' by {artist_name}."
        
        video = search_response['items'][0]
        video_id = video['id']['videoId']
        video_title = video['snippet']['title']
        channel_title = video['snippet']['channelTitle']
        
        # Return in a format that signals we want to embed the video
        return f"YOUTUBE_VIDEO|{video_id}|{video_title}|{channel_title}"
        
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"


# Export all tools
ALL_TOOLS = [
    # Customer account tools
    get_customer_account,
    get_invoice_history,
    get_purchased_tracks,
    get_spending_summary,
    get_invoice_details,
    # Music catalog tools
    search_tracks,
    search_artists,
    get_artist_albums,
    get_album_tracks,
    get_genres,
    get_tracks_by_genre,
    get_popular_tracks,
    get_track_price,
    # Lyrics search tools
    search_song_by_lyrics,
    check_song_in_catalogue,
    search_youtube_video,
]
