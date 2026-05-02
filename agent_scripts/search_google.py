import requests
from bs4 import BeautifulSoup
import urllib.parse

"""A script to perform a Google search and extract the title/name of the first search result."""

def run(params):
    # Construct the search URL using the provided parameters
    if "query" not in params or not params["query"]:
        return "Error: Query parameter is missing."

    # Google requires the query parameter 'q'
    search_url = f"{google_base_url}?q={urllib.parse.quote(params['query'])}"

    print(f"Attempting to fetch Google search results from: {search_url}")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    
        soup = BeautifulSoup(response.content, 'html.parser')
    
        # We will look for the main link title element.
        first_result_title = soup.find('h3')
    
        if first_result_title:
            return first_result_title.get_text(strip=True)
        else:
            return "Could not find the name of the first result. Google's structure requires a dedicated API key for reliable scraping or the selector needs updating."
    except requests.exceptions.RequestException as e:
        return f"An error occurred during the request: {e}"
    except Exception as e:
        return f"An unexpected error occurred during parsing: {e}"


if __name__ == "__main__":
    # The base URL is a fixed constant for Google searches.
    google_base_url = "https://www.google.com/search"
    params = {"query": "iphone"}
    run(params)
