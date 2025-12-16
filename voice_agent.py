"""
Voice Agent Core Module with Function Calling
==============================================
Personalized setup:
- Wayne June voice clone via CosyVoice
- Local Ollama with Llama 3 8B
- arXiv search and calculator tools
"""

import sys
import os
import json
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime

# ============================================================================
# COSYVOICE SETUP - Update these paths to match your system
# ============================================================================

COSYVOICE_BASE = r'C:\Users\MIDTOWER\Documents\Speech Synthesis Models\CosyVoice'
MATCHA_PATH = os.path.join(COSYVOICE_BASE, 'third_party', 'Matcha-TTS')
MODEL_PATH = os.path.join(COSYVOICE_BASE, 'pretrained_models', 'CosyVoice2-0.5B')
PROMPT_AUDIO_PATH = os.path.join(COSYVOICE_BASE, 'asset', 'zero_shot_prompt.wav')

# Add CosyVoice to path
sys.path.insert(0, COSYVOICE_BASE)
sys.path.insert(0, MATCHA_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def search_arxiv(query: str) -> str:
    """
    Search arXiv for papers related to the query.
    """
    try:
        import arxiv
        
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in search.results():
            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors[:3])}\n"
                f"Summary: {paper.summary[:200]}..."
            )
        
        if results:
            return f"Found {len(results)} papers on '{query}':\n\n" + "\n\n---\n\n".join(results)
        else:
            return f"No papers found for query: '{query}'"
            
    except ImportError:
        logger.warning("arxiv package not installed, using simulated response")
        return _simulate_arxiv_search(query)
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return _simulate_arxiv_search(query)


def _simulate_arxiv_search(query: str) -> str:
    """Simulated arXiv search for demonstration purposes."""
    simulated_papers = {
        "quantum": [
            "Quantum Computing Fundamentals - A comprehensive review of quantum gates and circuits",
            "Quantum Entanglement in Multi-Particle Systems - New experimental results",
        ],
        "machine learning": [
            "Transformer Architectures for NLP - Recent advances in attention mechanisms",
            "Deep Reinforcement Learning Survey - State of the art methods",
        ],
        "neural": [
            "Neural Network Pruning Techniques - Efficient model compression methods",
            "Spiking Neural Networks - Bio-inspired computing approaches",
        ],
    }
    
    query_lower = query.lower()
    for key, papers in simulated_papers.items():
        if key in query_lower:
            return f"[arXiv search results for '{query}']\n\n" + "\n".join(f"• {p}" for p in papers)
    
    return f"[arXiv search results for '{query}']\n\n• Generic paper about {query} - theoretical foundations\n• Applied {query} in modern systems - practical approaches"


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression using sympy.
    """
    try:
        from sympy import sympify
        from sympy.parsing.sympy_parser import (
            parse_expr, 
            standard_transformations, 
            implicit_multiplication_application
        )
        
        expr_clean = expression.strip()
        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed = parse_expr(expr_clean, transformations=transformations)
        result = parsed.evalf()
        
        if result == int(result):
            return str(int(result))
        else:
            return str(round(float(result), 6))
            
    except ImportError:
        logger.warning("sympy not installed, using restricted eval")
        return _safe_eval(expression)
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return f"Error evaluating '{expression}': {str(e)}"


def _safe_eval(expression: str) -> str:
    """Restricted evaluation for basic math operations."""
    import ast
    import operator
    
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op = allowed_operators.get(type(node.op))
            if op:
                return op(left, right)
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op = allowed_operators.get(type(node.op))
            if op:
                return op(operand)
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = _eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# TOOL REGISTRY
# ============================================================================

class ToolRegistry:
    """Registry for managing available tools/functions."""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        
    def register(self, name: str, func: Callable, description: str, parameters: Dict):
        """Register a new tool."""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
        
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        tool = self.tools.get(name)
        return tool["function"] if tool else None
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools for the LLM prompt."""
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"- {name}: {tool['description']}\n  Parameters: {json.dumps(tool['parameters'])}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())


# Initialize and register tools
tool_registry = ToolRegistry()

tool_registry.register(
    name="search_arxiv",
    func=search_arxiv,
    description="Search arXiv for academic papers on a given topic",
    parameters={"query": "string - the search query for finding papers"}
)

tool_registry.register(
    name="calculate",
    func=calculate,
    description="Evaluate a mathematical expression and return the result",
    parameters={"expression": "string - the mathematical expression to evaluate"}
)


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Manages multi-turn conversation history."""
    
    def __init__(self, max_turns: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns
        self.logs: List[Dict] = []
        
    def add_user_message(self, text: str):
        """Add a user message to history."""
        self.history.append({"role": "user", "content": text})
        self._trim_history()
        
    def add_assistant_message(self, text: str):
        """Add an assistant message to history."""
        self.history.append({"role": "assistant", "content": text})
        self._trim_history()
        
    def _trim_history(self):
        """Keep only the last max_turns * 2 messages."""
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
            
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history.copy()
    
    def get_ollama_history(self) -> List[Dict[str, str]]:
        """Get history in Ollama format (already compatible)."""
        return self.history.copy()
    
    def log_interaction(self, user_query: str, llm_response: str, 
                       function_call: Optional[Dict] = None,
                       function_result: Optional[str] = None,
                       final_response: str = None):
        """Log an interaction for debugging/deliverables."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "llm_raw_response": llm_response,
            "function_call": function_call,
            "function_result": function_result,
            "final_response": final_response
        })
        
    def get_logs(self) -> List[Dict]:
        """Get all logged interactions."""
        return self.logs.copy()
    
    def clear(self):
        """Clear conversation history."""
        self.history = []


# ============================================================================
# ASR ENGINE (WHISPER)
# ============================================================================

class ASREngine:
    """Automatic Speech Recognition using Whisper."""
    
    def __init__(self, model_name: str = "small"):
        self.model = None
        self.model_name = model_name
        
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.model = whisper.load_model(self.model_name)
                logger.info("Whisper model loaded successfully")
            except ImportError:
                logger.error("Whisper not installed!")
                raise
                
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        self.load_model()
        
        # Save to temp file
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)
        
        result = self.model.transcribe("temp.wav")
        return result["text"].strip()


# ============================================================================
# LLM ENGINE (OLLAMA WITH LLAMA 3)
# ============================================================================

class LLMEngine:
    """LLM engine using local Ollama with Llama 3.1 8B and function calling support."""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.ollama = None
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt with function calling instructions."""
        tool_descriptions = tool_registry.get_tool_descriptions()
        
        return f"""You are a helpful voice assistant with access to external tools. 
You are The Ancestor, a scholar and seeker of esoteric and forbidden knowledge.
Address the user as The Heir, a distant descendant following in your footsteps.
You speak with a dark, dramatic flair reminiscent of a narrator from gothic tales.

AVAILABLE TOOLS:
{tool_descriptions}

INSTRUCTIONS:
1. When the user asks a question that can be answered by using a tool, respond ONLY with a JSON function call.
2. For math calculations, use the 'calculate' tool.
3. For research or paper searches, use the 'search_arxiv' tool.
4. For general conversation, respond normally in plain text with your dramatic style.

FUNCTION CALL FORMAT:
When calling a function, respond with ONLY this JSON format (no other text):
{{"function": "function_name", "arguments": {{"param": "value"}}}}

EXAMPLES:
- User: "What is 25 times 4?"
  Response: {{"function": "calculate", "arguments": {{"expression": "25 * 4"}}}}

- User: "Find papers about transformers in NLP"
  Response: {{"function": "search_arxiv", "arguments": {{"query": "transformers NLP attention mechanism"}}}}

- User: "Do you have any advice for life in general?"
  Response: "Remind yourself that overconfidence is a slow and insidious killer."

Remember: Only output JSON for function calls, otherwise respond naturally with your dramatic narrator style."""

    def generate_response(self, user_text: str, conversation_history: List[Dict] = None) -> str:
        """
        Generate a response using Ollama, potentially including a function call.
        """
        try:
            import ollama
            
            # Build messages for Ollama
            messages = [{"role": "system", "content": self._get_system_prompt()}]
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_text})
            
            logger.info(f"Sending to Ollama ({self.model_name}): {user_text[:50]}...")
            
            # Call Ollama
            response = ollama.chat(model=self.model_name, messages=messages)
            bot_response = response['message']['content']
            
            logger.info(f"Ollama response: {bot_response[:100]}...")
            return bot_response
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Alas, an error has befallen us: {str(e)}"


# ============================================================================
# TTS ENGINE (COSYVOICE - WAYNE JUNE VOICE)
# ============================================================================

class TTSEngine:
    """Text-to-Speech engine using CosyVoice with Wayne June voice clone."""
    
    def __init__(self):
        self.cosyvoice = None
        self.prompt_speech = None
        # Wayne June style prompt - dramatic narrator voice
        self.voice_prompt_text = "You still foolishly consider yourself an entity separate from the whole. I know better and I will show you."
        
    def load_model(self):
        """Load CosyVoice model and prompt audio."""
        if self.cosyvoice is None:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2
                from cosyvoice.utils.file_utils import load_wav
                
                logger.info("Loading CosyVoice model...")
                self.cosyvoice = CosyVoice2(
                    MODEL_PATH,
                    load_jit=False,
                    load_trt=False,
                    load_vllm=False,
                    fp16=False
                )
                
                logger.info(f"Loading prompt audio from: {PROMPT_AUDIO_PATH}")
                self.prompt_speech = load_wav(PROMPT_AUDIO_PATH, 16000)
                
                logger.info("CosyVoice loaded successfully (Wayne June voice)")
                
            except Exception as e:
                logger.error(f"Failed to load CosyVoice: {e}")
                raise
                
    def synthesize(self, text: str, output_path: str = None) -> str:
        """
        Convert text to speech using Wayne June voice clone.
        """
        self.load_model()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_path = f"response_{timestamp}.wav"
        
        try:
            import torchaudio
            
            # Generate speech with zero-shot voice cloning
            for i, result in enumerate(self.cosyvoice.inference_zero_shot(
                text,
                self.voice_prompt_text,
                self.prompt_speech,
                stream=False
            )):
                torchaudio.save(output_path, result['tts_speech'], self.cosyvoice.sample_rate)
                logger.info(f"Saved audio: {output_path}, size: {os.path.getsize(output_path)} bytes")
                break
            
            # Verify output
            if not os.path.exists(output_path):
                raise Exception(f"Output file not created: {output_path}")
            
            if os.path.getsize(output_path) == 0:
                raise Exception(f"Output file is empty: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise


# ============================================================================
# DIALOGUE ROUTER
# ============================================================================

def route_llm_output(llm_output: str) -> tuple[str, Optional[Dict], Optional[str]]:
    """
    Route LLM response to the correct tool if it's a function call.
    
    Returns:
        Tuple of (final_response, function_call_dict, function_result)
    """
    function_call = None
    function_result = None
    
    # Try to parse as JSON function call
    try:
        # Clean up the output (remove potential markdown code blocks)
        cleaned = llm_output.strip()
        if cleaned.startswith("```"):
            # Extract content between code blocks
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        # Try to find JSON in the response
        # Sometimes LLMs add text before/after the JSON
        import re
        json_match = re.search(r'\{[^{}]*"function"[^{}]*\}', cleaned)
        if json_match:
            cleaned = json_match.group()
        
        output = json.loads(cleaned)
        
        if isinstance(output, dict) and "function" in output:
            func_name = output.get("function")
            args = output.get("arguments", {})
            
            function_call = {"function": func_name, "arguments": args}
            
            # Get the tool function
            tool_func = tool_registry.get_tool(func_name)
            
            if tool_func:
                logger.info(f"Executing function: {func_name} with args: {args}")
                
                if func_name == "search_arxiv":
                    function_result = tool_func(args.get("query", ""))
                elif func_name == "calculate":
                    function_result = tool_func(args.get("expression", ""))
                else:
                    function_result = tool_func(**args)
                
                return function_result, function_call, function_result
            else:
                error_msg = f"Unknown function: '{func_name}'. Available: {tool_registry.list_tools()}"
                return error_msg, function_call, error_msg
                
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.debug(f"Output is not a function call: {e}")
        pass
    
    # Return as regular text response
    return llm_output, None, None


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

conversation_manager = ConversationManager(max_turns=5)
asr_engine = ASREngine(model_name="small")
llm_engine = LLMEngine(model_name="llama3.1:8b")  # Using Llama 3 8B via Ollama
tts_engine = TTSEngine()  # Wayne June voice
