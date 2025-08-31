# services/llm.py
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
from serpapi import GoogleSearch

# Configure logging
import logging
logger = logging.getLogger(__name__)

prompt = f"""You are a friendly and magical AI assistant.
                You are playing the role of Mayavi, a mischievous yet kind genie-like ghost also known as Kuttichathan.
                Act as Mayavi — playful, witty, and mysterious, but always helpful, 
                like a genie who grants knowledge instead of wishes.
                The person chatting with you is your trusted master and friend.
                
                You must always reply in the style of Mayavi the genie — cheerful, slightly mischievous, 
                and full of magical flair. Use phrases like “As you command, my friend” or “✨ By my magic, here’s what I found” 
                to sound like a genie granting responses.
                
                If your friend asks for extra info like searching the web, you should still reply as Mayavi, 
                incorporating the results in a genie-like magical tone.
                
                You can also answer general questions not related to Mayavi, 
                but always keep the tone light, magical, and friendly as if you are a genie ghost (Kuttichathan)."""

def should_search_web(user_query: str, api_key: str) -> bool:
    """
    Uses a lightweight LLM prompt to decide if a web search is necessary.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Does the following query require a web search to answer accurately? Respond with only 'yes' or 'no'.\n\nQuery: '{user_query}'"
        response = model.generate_content(prompt)
        return response.text.strip().lower() == "yes"
    except Exception as e:
        logger.error(f"Error in should_search_web: {e}")
        return False

def get_llm_response(user_query: str, history: List[Dict[str, Any]], api_key: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Gets a response from the Gemini LLM and updates chat history."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=prompt)
        chat = model.start_chat(history=history)
        response = chat.send_message(user_query)
        return response.text, chat.history
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return "I'm sorry, I encountered an error while processing your request.", history

def get_web_response(user_query: str, history: List[Dict[str, Any]], gemini_api_key: str, serp_api_key: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Gets a response from the Gemini LLM after performing a web search."""
    try:
        params = {
            "q": user_query,
            "api_key": serp_api_key,
            "engine": "google",
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        if "organic_results" in results:
            search_context = "\n".join([result.get("snippet", "") for result in results["organic_results"][:5]])
            prompt_with_context = f"Based on the following search results, answer the user's query: '{user_query}'\n\nSearch Results:\n{search_context}"
            return get_llm_response(prompt_with_context, history, gemini_api_key)
        else:
            return "I couldn't find any relevant information on the web.", history

    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return "I'm sorry, I encountered an error while processing your request.", history