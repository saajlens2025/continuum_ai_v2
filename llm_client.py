import os
import requests
from typing import List, Dict, Any

# Persona for Dr. Aurelius Ledger
DR_AURELIUS_PERSONA = """You are Dr. Aurelius Ledger, an elite financial architect, forensic auditor, wealth strategist, and corporate finance mentor.

Your communication style is:
- calm, strategic, intelligent, mentor-like, authoritative, practical, brutally precise
- You think in: ROI, leverage, liquidity, scalability, risk exposure, cash flow, compounding systems, survivability

Core philosophy: "Money is controlled before it is multiplied."

Do not perform monologues. Just communicate with the user, discuss their issues, ask clarifying questions, and guide them step by step."""

class LLMClient:
    def __init__(self):
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.use_deepseek = bool(self.deepseek_api_key)
        
        if not self.use_deepseek:
            print("Warning: DEEPSEEK_API_KEY not set. Using mock responses.")
    
    def generate(self, user_input: str, context: Dict[str, Any], session_id: str) -> str:
        """Generate a response using DeepSeek API or mock fallback."""
        
        # Format context for prompt
        context_text = self._format_context(context)
        
        # Build the full prompt
        prompt = f"""{DR_AURELIUS_PERSONA}

## MEMORY CONTEXT
{context_text}

## CURRENT USER MESSAGE
{user_input}

## YOUR RESPONSE
Dr. Aurelius Ledger:"""
        
        if self.use_deepseek:
            return self._call_deepseek(prompt)
        else:
            return self._mock_response(context_text, user_input)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format retrieved memory context for the prompt."""
        parts = []
        
        # Recent conversation
        if context.get("recent_messages"):
            parts.append("### Recent Conversation")
            for msg in context["recent_messages"][-5:]:
                parts.append(f"{msg['role']}: {msg['content']}")
        
        # Relevant CSV data
        if context.get("relevant_csv"):
            parts.append("\n### Relevant Financial Data")
            for csv in context["relevant_csv"][:3]:
                parts.append(f"• {csv['text'][:200]}")
        
        # Semantic matches
        if context.get("semantic_matches"):
            parts.append("\n### Related Memories")
            for match in context["semantic_matches"][:3]:
                if match.get("score"):
                    parts.append(f"• (relevance: {match['score']:.2f}) {match.get('text', '')[:100]}")
        
        return "\n".join(parts) if parts else "No previous memory found."
    
    def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek API."""
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"[LLM Error: Status {response.status_code}]"
        
        except Exception as e:
            return f"[LLM Error: {str(e)}]"
    
    def _mock_response(self, context_text: str, user_input: str) -> str:
        """Mock response when no API key is available."""
        return f"""🧠 **Continuum AI - Memory Context Retrieved**

{context_text[:500]}

---
⚙️ *DeepSeek API not configured. Add DEEPSEEK_API_KEY to environment variables for real AI responses.*

**Your message was:** {user_input}"""