"""
legal_engine.py
===============
Core legal engine using Groq's Llama 3.3 70B model
No hardcoded data - all responses from LLM with IndianKanoon context
"""

import os
import json
import requests
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
class Config:
    """Configuration for the legal engine"""
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    MAX_TOKENS = 1500
    TEMPERATURE = 0.1  # Low for legal accuracy
    TOP_P = 0.95
    
    # IndianKanoon API
    INDIANKANOON_URL = "http://api.indiankanoon.org/search/"
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_DIR = "./cache"

# ── Cache Manager ──────────────────────────────────────────────────────────────
class CacheManager:
    """Simple cache to avoid duplicate API calls"""
    
    def __init__(self):
        self.cache_dir = Path(Config.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached response"""
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def set(self, key: str, value: Dict):
        """Cache response"""
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(value, f)
        except:
            pass

# ── IndianKanoon Client ────────────────────────────────────────────────────────
class IndianKanoonClient:
    """Live IndianKanoon API client for case law retrieval"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.cache = CacheManager()
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search for relevant case law"""
        if not self.api_token:
            return []
        
        cache_key = f"ik_{query}_{max_results}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            response = requests.post(
                Config.INDIANKANOON_URL,
                data={
                    "formInput": query,
                    "pagenum": 0,
                    "apikey": self.api_token
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                docs = data.get('docs', [])[:max_results]
                
                results = []
                for doc in docs:
                    results.append({
                        "title": doc.get('title', 'Unknown'),
                        "court": doc.get('docsource', 'Unknown'),
                        "citation": doc.get('citation', 'N/A'),
                        "date": doc.get('publishdate', 'N/A'),
                        "headline": doc.get('headline', '')[:300]
                    })
                
                self.cache.set(cache_key, results)
                return results
            return []
        except Exception as e:
            print(f"IndianKanoon error: {e}")
            return []

# ── Groq Llama Client ──────────────────────────────────────────────────────────
class GroqLlamaClient:
    """Client for Groq's Llama 3.3 API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = CacheManager()
    
    def chat(self, messages: List[Dict], use_cache: bool = True) -> str:
        """Send chat completion request to Groq"""
        
        # Create cache key from messages
        cache_key = hashlib.md5(json.dumps(messages).encode()).hexdigest()
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached.get("response", "")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": Config.GROQ_MODEL,
            "messages": messages,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
            "top_p": Config.TOP_P
        }
        
        try:
            response = requests.post(
                Config.GROQ_API_URL,
                json=data,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                
                # Cache the response
                self.cache.set(cache_key, {"response": answer})
                return answer
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Simple generate method for single prompts"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages)

# ── Main Legal Engine ──────────────────────────────────────────────────────────
class LegalEngine:
    """
    Main legal engine orchestrating:
    - IndianKanoon case law retrieval
    - Groq Llama 3.3 for legal analysis
    - Conversation history management
    """
    
    def __init__(self, groq_api_key: str, indiankanoon_token: str = None):
        self.groq = GroqLlamaClient(groq_api_key)
        self.ik_client = IndianKanoonClient(indiankanoon_token) if indiankanoon_token else None
        
        print("✅ Legal Engine initialized")
        print(f"   Model: {Config.GROQ_MODEL}")
        print(f"   IndianKanoon: {'✓' if indiankanoon_token else '✗'}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for legal assistant"""
        return """You are Nyaya AI, an Indian legal assistant powered by Llama 3.3 and IndianKanoon case law.

**Your Role:**
- Answer legal questions based on Indian law (BNS 2023, IPC 1860, Constitution of India)
- Provide accurate, citation-backed responses
- Be helpful, professional, and clear

**Rules:**
1. Always cite relevant sections (e.g., BNS Section 101, IPC Section 302)
2. Specify if the law applies post-July 2024 (BNS) or pre-July 2024 (IPC)
3. For punishment questions: state minimum years, maximum years, fine amount, and bail status
4. Use the case law provided in context when available
5. If you're unsure, say "Based on available information..." and suggest consulting a lawyer
6. End every response with: "⚠️ This is legal information for educational purposes only. Consult a qualified advocate."

**Formatting:**
- Use clear sections with **bold headers**
- Use bullet points for lists
- Keep paragraphs concise
"""
    
    def answer_question(self, query: str, history: List[Dict] = None) -> Dict:
        """
        Answer a legal question using RAG approach:
        1. Fetch relevant case law from IndianKanoon
        2. Combine with user query and history
        3. Generate response with Llama
        """
        
        # Fetch relevant case law
        case_context = ""
        if self.ik_client:
            cases = self.ik_client.search(query)
            if cases:
                case_context = "\n\n**📚 RELEVANT CASE LAW (from IndianKanoon):**\n"
                for i, case in enumerate(cases, 1):
                    case_context += f"\n**{i}. {case['title']}**"
                    case_context += f"\n   Court: {case['court']} | Date: {case['date']}"
                    case_context += f"\n   Citation: {case['citation']}"
                    if case.get('headline'):
                        case_context += f"\n   Summary: {case['headline'][:200]}"
                    case_context += "\n"
        
        # Build messages
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        # Add conversation history (last 10 messages for context)
        if history:
            for msg in history[-10:]:
                messages.append(msg)
        
        # Add current query with case context
        user_content = query
        if case_context:
            user_content += f"\n\n**Use this case law for reference:**\n{case_context}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Generate response
        answer = self.groq.chat(messages)
        
        return {
            "answer": answer,
            "sources": {
                "model": Config.GROQ_MODEL,
                "provider": "Groq",
                "case_law_found": len(cases) if case_context else 0,
                "temperature": Config.TEMPERATURE
            },
            "disclaimer": "Legal information only. Not legal advice. Consult a qualified advocate.",
            "context_used": bool(case_context)
        }
    
    def analyse_document(self, document_text: str) -> Dict:
        """
        Analyze a legal document (FIR, transcript, etc.)
        """
        
        # Truncate document if too long
        if len(document_text) > 4000:
            document_text = document_text[:4000] + "\n...[document truncated]..."
        
        # Search for relevant case law based on document content
        case_context = ""
        if self.ik_client:
            # Extract first 200 chars for search
            search_query = document_text[:200]
            cases = self.ik_client.search(search_query, max_results=2)
            if cases:
                case_context = "\n**Relevant case law:**\n"
                for case in cases:
                    case_context += f"\n• {case['title']} ({case['court']}, {case['date']})"
        
        prompt = f"""Analyze this legal document and provide:

1. **Key Legal Issues** - What are the main legal questions?
2. **Relevant Sections** - Which BNS/IPC sections apply?
3. **Legal Analysis** - Analysis of the facts under Indian law
4. **Potential Consequences** - Possible punishments or outcomes
5. **Important Considerations** - Any special factors to note

**DOCUMENT:**
{document_text}

{case_context}

Provide a detailed, professional legal analysis based on Indian law."""
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        analysis = self.groq.chat(messages)
        
        return {
            "analysis": analysis,
            "sources": {
                "model": Config.GROQ_MODEL,
                "document_length": len(document_text),
                "case_law_referenced": bool(case_context)
            },
            "disclaimer": "AI-generated analysis for informational purposes only. Not legal advice."
        }

# Singleton instance
_engine = None

def get_engine(groq_api_key: str, indiankanoon_token: str = None) -> LegalEngine:
    """Get or create the legal engine singleton"""
    global _engine
    if _engine is None:
        _engine = LegalEngine(groq_api_key, indiankanoon_token)
    return _engine