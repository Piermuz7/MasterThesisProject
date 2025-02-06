from duckduckgo_search import DDGS


async def search_web(query: str, num_results=5):
    "Perform a web search using DuckDuckGo and return relevant results."
    results = []

    with DDGS() as ddgs:
        search_results = ddgs.text(query, max_results=num_results)

    for res in search_results:
        results.append({
            "title": res["title"],
            "url": res["href"],
            "snippet": res.get("body", "No description available.")
        })

    return results
