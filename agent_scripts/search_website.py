import requests
from bs4 import BeautifulSoup
import urllib.parse

"""A script to search a given website for a query and extract the name of the first result."""

def run(params):
    # Construct the search URL using the provided parameters
    if "query" not in params or not params["query"]:
        return "Error: Query parameter is missing."

    # Correctly URL encode the query
    encoded_query = urllib.parse.quote(params["query"])
    base_url = "https://www.brack.ch/search"
    search_url = f"{base_url}?query={encoded_query}"

    print(f"Attempting to fetch search results from: {search_url}")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    
        soup = BeautifulSoup(response.content, 'html.parser')
    
        # --- SCALING NOTE ---
        # This selector remains a placeholder. A real inspection of brack.ch is required.
        # For testing, we will try to find a generic product title selector.
        first_result = soup.find('h2', class_='result-title') # Placeholder selector
    
        if first_result:
            return first_result.get_text(strip=True)
        else:
            # Attempting a more general selector just in case
            generic_result = soup.find('h3', class_=lambda c: c and ('product' in c or 'title' in c))
            if generic_result:
                 return generic_result.get_text(strip=True)
            return "Could not find the name of the first result. The website structure may have changed, or the selector needs adjustment."
    except requests.exceptions.RequestException as e:
        return f"An error occurred during the request: {e}"
    except Exception as e:
        return f"An unexpected error occurred during parsing: {e}"


if __name__ == "__main__":
    # The base URL should be 'https://www.brack.ch/search'
    # The parameter name should be 'query'
    pass
    params = {"query": "iphone x"}
    run(params)
