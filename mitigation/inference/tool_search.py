from typing import List, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool
from youdotcom import You
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        api_key = os.getenv("YOU_API_KEY") or os.getenv("YDC_API_KEY")
        if not api_key:
            raise ValueError("YOU_API_KEY or YDC_API_KEY environment variable must be set")
        self.you_client = You(api_key)
    
    def you_search_single(self, query: str, max_results: int = 10):
        """Perform a single you.com search query."""
        try:
            result = self.you_client.search.unified(query=query)
            
            if not result:
                return f"No results found for '{query}'. Try with a more general query."
            
            # Helper function to safely get value from dict or object
            def get_value(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    return getattr(obj, key, default) if hasattr(obj, key) else default
            
            # Get results - handle both dict and object responses
            results_obj = get_value(result, 'results', None)
            if not results_obj:
                return f"No results found for '{query}'. Try with a more general query."
            
            web_snippets = []
            results_list = []
            
            # Get web results
            web_results = get_value(results_obj, 'web', [])
            if web_results:
                results_list.extend(web_results[:max_results])
            
            # Get news results if available
            news_results = get_value(results_obj, 'news', [])
            if news_results and len(results_list) < max_results:
                remaining = max_results - len(results_list)
                results_list.extend(news_results[:remaining])
            
            if not results_list:
                return f"No results found for '{query}'. Try with a more general query."
            
            for idx, page in enumerate(results_list, 1):
                title = get_value(page, 'title', 'No title')
                url = get_value(page, 'url', '')
                description = get_value(page, 'description', '')
                snippets = get_value(page, 'snippets', [])
                
                # Combine description and snippets for content
                content_parts = [description] if description else []
                if snippets:
                    if isinstance(snippets, list):
                        content_parts.extend(snippets)
                    else:
                        content_parts.append(str(snippets))
                content = '\n'.join(content_parts) if content_parts else "No content available"
                
                redacted_version = f"{idx}. [{title}]({url})\n{content}"
                web_snippets.append(redacted_version)
            
            formatted_result = f"A search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return formatted_result
            
        except Exception as e:
            print(f"You.com search error: {e}")
            return f"Search failed for '{query}'. Error: {str(e)}"

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # Single query
            response = self.you_search_single(query)
        else:
            # Multiple queries
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.you_search_single(q))
            response = "\n=======\n".join(responses)
            
        return response

