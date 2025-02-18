from googlesearch import search

async def search_web(query: str, num_results: int = 5):
    """Perform a web search using Google Search and return relevant results."""
    results = []

    # Perform search with advanced=True to get more metadata
    search_results = search(query, advanced=True, num_results=num_results)

    for res in search_results:
        results.append({
            "title": res.title,
            "url": res.url,
            "snippet": res.description if res.description else "No description available."
        })

    return results