"""
Core Web Search Functions for F.R.E.D. V2
Modular, robust web search system with spam filtering and intelligent routing.
"""

import json
import re
import requests
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
import trafilatura
from datetime import datetime

import config
from ollama_manager import OllamaConnectionManager
from Tools import search_brave, search_searchapi  # Import existing search functions

# Initialize Ollama manager
ollama_manager_instance = OllamaConnectionManager(config.OLLAMA_BASE_URL, config.LLM_GENERATION_OPTIONS)
ollama_manager = ollama_manager_instance

# Domain blacklist for spam/ad filtering
SPAM_DOMAINS = {
    'pinterest.com', 'pinterest.ca', 'pinterest.co.uk',
    'facebook.com', 'twitter.com', 'instagram.com',
    'youtube.com', 'tiktok.com', 'snapchat.com',
    'reddit.com/r/*/comments',  # Individual Reddit posts (too specific)
    'quora.com/What-',  # Quora question pages (often low quality)
    'answers.yahoo.com', 'wiki.answers.com',
    'ehow.com', 'wikihow.com',
    'scribd.com', 'slideshare.net',
    'amazon.com/dp/', 'amazon.com/gp/product/',  # Product pages
    'ebay.com/itm/', 'aliexpress.com/item/',
    'shopify.com', 'etsy.com/listing/',
    'medium.com/@', # Personal Medium blogs (often low quality)
}

# URL pattern filters for spam/ad content
SPAM_URL_PATTERNS = [
    r'/buy-now/', r'/purchase/', r'/checkout/',
    r'/affiliate/', r'/referral/', r'/promo/',
    r'/discount/', r'/coupon/', r'/deal/',
    r'/signup/', r'/register/', r'/subscribe/',
    r'/download-now/', r'/free-trial/',
    r'\.pdf$',  # Skip direct PDF links in general search
    r'\.doc$', r'\.docx$', r'\.ppt$', r'\.pptx$',
]


def gather_links(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Gather links from web search using Brave → SearchAPI fallback.
    
    Args:
        query: Search query string
        max_results: Maximum number of links to return (default: 5)
        
    Returns:
        List of dictionaries with 'url', 'title', 'description' keys
    """
    all_results = []
    seen_urls = set()
    
    # Try Brave Search first
    try:
        brave_results = search_brave(query)
        if brave_results and isinstance(brave_results, str):
            # Parse Brave results (assuming it returns formatted string)
            # This would need to be adapted based on actual Brave function output
            for line in brave_results.split('\n'):
                if line.strip() and 'http' in line:
                    # Extract URL from Brave result line
                    url_match = re.search(r'https?://[^\s]+', line)
                    if url_match:
                        url = url_match.group()
                        title = line.split(' - ')[0] if ' - ' in line else "No title"
                        
                        if url not in seen_urls and not _is_spam_url(url):
                            all_results.append({
                                'url': url,
                                'title': title.strip(),
                                'description': line.replace(url, '').strip()
                            })
                            seen_urls.add(url)
                            
                            if len(all_results) >= max_results:
                                break
    except Exception as e:
        print(f"Brave search failed: {e}")
    
    # If we don't have enough results, try SearchAPI fallback
    if len(all_results) < max_results:
        try:
            searchapi_results = search_searchapi(query)
            if searchapi_results and isinstance(searchapi_results, str):
                # Parse SearchAPI results
                for line in searchapi_results.split('\n'):
                    if line.strip() and 'http' in line:
                        url_match = re.search(r'https?://[^\s]+', line)
                        if url_match:
                            url = url_match.group()
                            title = line.split(' - ')[0] if ' - ' in line else "No title"
                            
                            if url not in seen_urls and not _is_spam_url(url):
                                all_results.append({
                                    'url': url,
                                    'title': title.strip(),
                                    'description': line.replace(url, '').strip()
                                })
                                seen_urls.add(url)
                                
                                if len(all_results) >= max_results:
                                    break
        except Exception as e:
            print(f"SearchAPI fallback failed: {e}")
    
    return all_results[:max_results]


def extract_page_content(url: str) -> Optional[Dict[str, str]]:
    """
    Robustly extract readable content and metadata from a webpage.
    
    Args:
        url: URL to extract content from
        
    Returns:
        Dictionary with extracted content or None if extraction fails
        Keys: 'url', 'title', 'content', 'description', 'publish_date', 'author'
    """
    try:
        # Download the webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        html = response.text

        # Try extracting from the downloaded HTML first
        content = trafilatura.extract(html, include_comments=False, include_tables=True)
        metadata = trafilatura.extract_metadata(html) if content else None
        if not content:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
            content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if not content:
                return None
            metadata = trafilatura.extract_metadata(downloaded)
        
        # Build result dictionary
        result = {
            'url': url,
            'title': metadata.title if metadata and metadata.title else "No title",
            'content': content,
            'description': metadata.description if metadata and metadata.description else "",
            'publish_date': metadata.date if metadata and metadata.date else "",
            'author': metadata.author if metadata and metadata.author else ""
        }
        
        # Content-based spam filtering for extracted pages
        if _is_spam_content(result):
            return None
            
        return result
        
    except Exception as e:
        print(f"Failed to extract content from {url}: {e}")
        return None


def intelligent_search(query: str, search_priority: str = "quick", mode: str = "auto") -> Dict:
    """Perform a web search and rank links by semantic similarity.

    The function gathers links using the configured search engines and then
    ranks them using :func:`calculate_relevance_score`.  Links with the highest
    semantic similarity between the query and the page title are selected for
    content extraction.  This avoids the previous LLM-based link analysis and
    relies solely on embedding similarity.

    Args:
        query: Search query string
        search_priority: ``"quick"`` or ``"thorough"``
        mode: ``"links_only"``, ``"auto"``, or ``"deep"``

    Returns:
        Dictionary with search results and analysis
    """
    try:
        # Step 1: Gather initial links
        links = gather_links(query, max_results=8 if search_priority == "thorough" else 5)
        
        if mode == "links_only":
            return {
                'query': query,
                'links': links,
                'extracted_content': [],
                'summary': "Links only mode - no content extracted"
            }
        
        # Step 2: Rank links using semantic similarity
        scored_links = []
        for link in links:
            score = calculate_relevance_score(query, link.get('title', ''))
            link_with_score = link.copy()
            link_with_score['score'] = score
            scored_links.append(link_with_score)

        scored_links.sort(key=lambda x: x['score'], reverse=True)
        links = scored_links
        selected_urls = [item['url'] for item in links[:3]]
        
        # Step 3: Extract content from selected URLs
        extracted_content = []
        for url in selected_urls:
            content = extract_page_content(url)
            if content:
                extracted_content.append(content)
        
        # Step 4: Generate summary using GIST model
        if extracted_content:
            search_results_text = "\n\n".join([
                f"Title: {item['title']}\nURL: {item['url']}\nContent: {item['content'][:2000]}..."
                for item in extracted_content
            ])
            
            from prompts import GIST_SYSTEM_PROMPT, GIST_USER_PROMPT
            
            gist_messages = [
                {"role": "system", "content": GIST_SYSTEM_PROMPT},
                {"role": "user", "content": GIST_USER_PROMPT.format(
                    query=query,
                    search_results=search_results_text
                )}
            ]
            
            summary = ollama_manager.chat_concurrent_safe(
                model=config.GIST_OLLAMA_MODEL,
                messages=gist_messages,
                options=config.LLM_GENERATION_OPTIONS
            )
        else:
            summary = "No content could be extracted from the search results."
        
        return {
            'query': query,
            'links': links,
            'extracted_content': extracted_content,
            'summary': summary,
            'search_priority': search_priority,
            'mode': mode
        }
        
    except Exception as e:
        print(f"Intelligent search failed: {e}")
        return {
            'query': query,
            'links': [],
            'extracted_content': [],
            'summary': f"Search failed: {str(e)}",
            'error': str(e)
        }


def _is_spam_url(url: str) -> bool:
    """Check if URL matches spam/ad filtering criteria."""
    try:
        parsed = urlparse(url.lower())
        domain = parsed.netloc
        path = parsed.path
        
        # Check domain blacklist
        for spam_domain in SPAM_DOMAINS:
            if spam_domain in domain:
                return True
        
        # Check URL pattern filters
        full_url = f"{domain}{path}"
        for pattern in SPAM_URL_PATTERNS:
            if re.search(pattern, full_url):
                return True
                
        return False
        
    except:
        return True  # If parsing fails, assume it's spam


def _is_spam_content(content_dict: Dict[str, str]) -> bool:
    """Content-based spam filtering for extracted pages."""
    title = content_dict.get('title', '').lower()
    content = content_dict.get('content', '').lower()
    
    # Check for promotional language in title
    spam_title_keywords = [
        'buy now', 'discount', 'sale', 'coupon', 'deal',
        'free trial', 'sign up', 'register now',
        'click here', 'limited time', 'act now'
    ]
    
    for keyword in spam_title_keywords:
        if keyword in title:
            return True
    
    # Check content length and quality
    if len(content) < 200:  # Very short content likely spam
        return True
    
    # Check for excessive promotional content
    promo_count = sum(1 for keyword in ['buy', 'purchase', 'order', 'discount', 'sale'] if keyword in content)
    if promo_count > 5:  # Too many promotional keywords
        return True
        
    return False


# Add relevance scoring function for embedding-based ranking
def calculate_relevance_score(query: str, title: str) -> float:
    """
    Calculate relevance score between query and title using embeddings.
    
    Args:
        query: Search query
        title: Page title
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Get embeddings for both query and title
        query_embedding = ollama_manager.embeddings(model=config.EMBED_MODEL, prompt=query)
        title_embedding = ollama_manager.embeddings(model=config.EMBED_MODEL, prompt=title)
        
        if not query_embedding or not title_embedding:
            return 0.0
        
        # Calculate cosine similarity
        import numpy as np
        query_vec = np.array(query_embedding)
        title_vec = np.array(title_embedding)
        
        cosine_sim = np.dot(query_vec, title_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(title_vec))
        return max(0.0, cosine_sim)  # Ensure non-negative
        
    except Exception as e:
        print(f"Relevance calculation failed: {e}")
        return 0.0
