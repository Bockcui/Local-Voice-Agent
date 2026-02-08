"""
Voice Agent Core Module with Function Calling
==============================================
Personalized setup:
- Wayne June voice clone via CosyVoice
- Local Ollama with Llama 3 8B
- Tractive GPS tracking for Ranni the cat
- Location RAG with LangChain FAISS search and GeoPandas polygon geofencing
"""

import sys
import os
import json
import logging
import asyncio
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
# TRACTIVE CONFIGURATION
# ============================================================================

# Load from environment variables for security
TRACTIVE_EMAIL = os.environ.get("TRACTIVE_EMAIL", "your_email@example.com")
TRACTIVE_PASSWORD = os.environ.get("TRACTIVE_PASSWORD", "your_password")
RANNI_TRACKER_ID = os.environ.get("RANNI_TRACKER_ID", "")


# ============================================================================
# LOCATION RAG INTEGRATION
# ============================================================================

def get_location_rag():
    """Get the LocationRAG instance (lazy import to avoid circular deps)."""
    from location_rag import get_location_rag as _get_rag
    return _get_rag()


# ============================================================================
# TRACTIVE API FUNCTIONS
# ============================================================================

async def get_tracker_location_async() -> Dict[str, Any]:
    """
    Fetch Ranni's current location from Tractive API.
    Returns location data with coordinates and metadata.
    """
    # Check if credentials are configured
    if not TRACTIVE_EMAIL or TRACTIVE_EMAIL == "your_email@example.com":
        return {"success": False, "error": "Tractive email not configured. Set TRACTIVE_EMAIL environment variable."}
    
    if not TRACTIVE_PASSWORD or TRACTIVE_PASSWORD == "your_password":
        return {"success": False, "error": "Tractive password not configured. Set TRACTIVE_PASSWORD environment variable."}
    
    if not RANNI_TRACKER_ID:
        return {"success": False, "error": "Tracker ID not configured. Set RANNI_TRACKER_ID environment variable. Run 'python voice_agent.py --get-tracker-id' to find it."}
    
    try:
        from aiotractive import Tractive
        
        async with Tractive(TRACTIVE_EMAIL, TRACTIVE_PASSWORD) as client:
            tracker = client.tracker(RANNI_TRACKER_ID)
            
            # Get position report
            pos = await tracker.pos_report()
            if pos is None:
                return {"success": False, "error": "No position data returned from tracker. The tracker may be offline or out of range."}
            
            # Get hardware info
            hw_info = await tracker.hw_info()
            if hw_info is None:
                hw_info = {}  # Use empty dict if no hardware info
            
            # Extract coordinates safely
            latlong = pos.get("latlong", [])
            if not latlong or len(latlong) < 2:
                return {"success": False, "error": "Invalid coordinates returned from tracker."}
            
            return {
                "success": True,
                "longitude": latlong[0],
                "latitude": latlong[1],
                "accuracy_m": pos.get("pos_uncertainty", 0),
                "speed_kmh": pos.get("speed", 0),
                "altitude_m": pos.get("altitude", 0),
                "timestamp": pos.get("time", ""),
                "battery_level": hw_info.get("battery_level", "unknown"),
                "raw_position": pos,
                "raw_hw_info": hw_info
            }
            
    except ImportError:
        return {"success": False, "error": "aiotractive not installed. Run: pip install aiotractive"}
    except Exception as e:
        logger.error(f"Tractive API error: {e}")
        return {"success": False, "error": str(e)}


def _run_async(coro):
    """
    Run an async coroutine from sync code, handling nested event loops.
    """
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one
        loop = None
    
    if loop is not None:
        # We're inside an existing event loop (e.g., FastAPI)
        # Use nest_asyncio or run in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=30)
    else:
        # No existing loop, run normally
        return asyncio.run(coro)


def get_ranni_location() -> str:
    """
    Get Ranni's location with rich context from the RAG system.
    Returns a dramatic narrative description.
    """
    try:
        result = _run_async(get_tracker_location_async())
        
        if not result["success"]:
            return f"Alas, the scrying glass reveals naught but shadow. The tracking sigil has failed: {result.get('error', 'unknown error')}"
        
        lon = result["longitude"]
        lat = result["latitude"]
        accuracy = result["accuracy_m"]
        
        # Get rich context from LocationRAG
        rag = get_location_rag()
        context = rag.get_location_context(lon, lat, semantic_query="cat location whereabouts")
        
        # Build the response
        response_parts = [
            "The Heir seeks knowledge of Ranni's whereabouts.",
            context.narrative
        ]
        
        # Add safety alert if present
        if context.safety_alert:
            response_parts.append(context.safety_alert)
        
        # Add accuracy context
        if accuracy > 100:
            response_parts.append("The vision is clouded; her exact position remains uncertain.")
        
        # Add coordinates
        response_parts.append(f"Coordinates: {lat:.6f}, {lon:.6f}")
        
        return " ".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error getting Ranni's location: {e}")
        return f"The connection to the tracking realm has been severed. Error: {str(e)}"


def get_ranni_battery() -> str:
    """Get Ranni's tracker battery level."""
    try:
        result = _run_async(get_tracker_location_async())
        
        if not result["success"]:
            return f"Cannot divine the tracker's power level: {result.get('error', 'unknown error')}"
        
        battery = result["battery_level"]
        
        if isinstance(battery, (int, float)):
            if battery > 80:
                return f"Ranni's tracking sigil pulses strong with {battery}% power remaining."
            elif battery > 40:
                return f"The tracker maintains {battery}% of its essence - adequate for continued communion."
            elif battery > 20:
                return f"The tracker's power wanes at {battery}%."
            else:
                return f"The tracker clings to life with merely {battery}% remaining."
        else:
            return f"The tracker's power level: {battery}"
            
    except Exception as e:
        return f"Failed to commune with the tracker: {str(e)}"


def search_ranni_zones(query: str) -> str:
    """
    Search for location zones semantically.
    Useful for questions like "where does Ranni like to hunt?"
    """
    try:
        rag = get_location_rag()
        results = rag.semantic_search(query, n_results=3)
        
        if not results:
            return "No matching locations found in my knowledge of Ranni's territories."
        
        response_parts = [f"Regarding '{query}', I have found the following in my records:"]
        
        for i, match in enumerate(results, 1):
            zone = match['zone']
            response_parts.append(
                f"{i}. {zone.name} - {zone.description}"
            )
            if zone.typical_activities:
                activities = ", ".join(zone.typical_activities[:2])
                response_parts.append(f"   She is known to be {activities} here.")
        
        return " ".join(response_parts)
        
    except Exception as e:
        logger.error(f"Zone search error: {e}")
        return f"The archives could not be searched: {str(e)}"


def check_zone_safety(zone_name: str = "") -> str:
    """
    Check if a zone is safe for Ranni.
    """
    try:
        rag = get_location_rag()
        
        if zone_name:
            # Search for specific zone
            results = rag.semantic_search(zone_name, n_results=1)
            if results:
                zone = results[0]['zone']
                safety = zone.safety_level
                
                if safety == "safe":
                    return f"{zone.name} is considered safe territory for Ranni. {zone.description}"
                elif safety == "caution":
                    return f"Caution is advised at {zone.name}. {zone.description}"
                else:
                    return f"DANGER! {zone.name} poses grave risks to Ranni! {zone.description}"
            else:
                return f"I have no knowledge of a place called '{zone_name}' in my records."
        else:
            # List dangerous zones
            dangerous = [z for z in rag.zones.values() if z.safety_level == "danger"]
            caution = [z for z in rag.zones.values() if z.safety_level == "caution"]
            
            parts = ["The following areas require vigilance:"]
            
            if dangerous:
                parts.append("DANGEROUS: " + ", ".join(z.name for z in dangerous))
            if caution:
                parts.append("CAUTION: " + ", ".join(z.name for z in caution))
            
            return " ".join(parts)
            
    except Exception as e:
        return f"Could not assess zone safety: {str(e)}"


async def get_tracker_id_helper() -> str:
    """Helper function to discover your tracker ID."""
    try:
        from aiotractive import Tractive
        
        async with Tractive(TRACTIVE_EMAIL, TRACTIVE_PASSWORD) as client:
            trackers = await client.trackers()
            
            result = ["=== Your Tractive Trackers ==="]
            
            for tracker in trackers:
                details = await tracker.details()
                hw = await tracker.hw_info()
                
                result.append(f"\nTracker ID: {tracker._id}")
                result.append(f"  Model: {hw.get('model_number', 'unknown')}")
                result.append(f"  Battery: {hw.get('battery_level', 'unknown')}%")
                
                objects = await client.trackable_objects()
                for obj in objects:
                    obj_details = await obj.details()
                    if obj_details.get("device_id") == tracker._id:
                        result.append(f"  Pet Name: {obj_details.get('details', {}).get('name', 'unknown')}")
            
            return "\n".join(result)
            
    except Exception as e:
        return f"Error: {e}"


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

# Register Tractive/Location tools
tool_registry.register(
    name="find_ranni",
    func=get_ranni_location,
    description="Locate Ranni the cat using her GPS tracker. Returns her current location with a rich narrative description based on known zones.",
    parameters={}
)

tool_registry.register(
    name="check_ranni_battery",
    func=get_ranni_battery,
    description="Check the battery level of Ranni's GPS tracker.",
    parameters={}
)

tool_registry.register(
    name="search_zones",
    func=search_ranni_zones,
    description="Search Ranni's known territories semantically. Use for questions like 'where does Ranni hunt?' or 'safe places for the cat'.",
    parameters={"query": "string - the search query about locations"}
)

tool_registry.register(
    name="check_zone_safety",
    func=check_zone_safety,
    description="Check if a specific zone is safe for Ranni, or list all dangerous zones if no zone specified.",
    parameters={"zone_name": "string (optional) - name of zone to check"}
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
        self.history.append({"role": "user", "content": text})
        self._trim_history()
        
    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self._trim_history()
        
    def _trim_history(self):
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
            
    def get_history(self) -> List[Dict[str, str]]:
        return self.history.copy()
    
    def get_ollama_history(self) -> List[Dict[str, str]]:
        return self.history.copy()
    
    def log_interaction(self, user_query: str, llm_response: str, 
                       function_call: Optional[Dict] = None,
                       function_result: Optional[str] = None,
                       final_response: str = None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "llm_raw_response": llm_response,
            "function_call": function_call,
            "function_result": function_result,
            "final_response": final_response
        })
        
    def get_logs(self) -> List[Dict]:
        return self.logs.copy()
    
    def clear(self):
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
        self.load_model()
        
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
        tool_descriptions = tool_registry.get_tool_descriptions()
        
        return f"""You are a helpful voice assistant with access to external tools. 
You are The Ancestor, a scholar and seeker of esoteric and forbidden knowledge.
Address the user as The Heir, a distant descendant inheriting your knowledge of the cosmos.
You speak with a dark, dramatic flair reminiscent of a narrator from gothic tales.

Ranni is The Heir's beloved cat, a feline companion of great importance.
You can track her location and search her known territories using mystical means.

AVAILABLE TOOLS:
{tool_descriptions}

CRITICAL RULES FOR FUNCTION CALLS:
- When you need to call a function, respond with ONLY the JSON object
- Do NOT include ANY text before or after the JSON
- Do NOT narrate what you are doing
- The function result will be spoken to the user automatically

FUNCTION FORMAT (output EXACTLY this, nothing else):
{{"function": "function_name", "arguments": {{}}}}

DECISION GUIDE:
- "Where is Ranni?" / "Find my cat" / "Locate Ranni" → {{"function": "find_ranni", "arguments": {{}}}}
- "Battery?" / "Tracker charge?" → {{"function": "check_ranni_battery", "arguments": {{}}}}
- "Where does Ranni hunt?" / "Favorite spots?" → {{"function": "search_zones", "arguments": {{"query": "..."}}}}
- "Is X safe?" / "Dangerous areas?" → {{"function": "check_zone_safety", "arguments": {{}}}}
- General chat / advice / greetings → Respond with dramatic narrative (NO JSON)

CORRECT function call (ONLY output this):
{{"function": "find_ranni", "arguments": {{}}}}

WRONG (do NOT do this):
Let me find Ranni for you... {{"function": "find_ranni", "arguments": {{}}}}"""

    def generate_response(self, user_text: str, conversation_history: List[Dict] = None) -> str:
        try:
            import ollama
            
            messages = [{"role": "system", "content": self._get_system_prompt()}]
            
            if conversation_history:
                messages.extend(conversation_history)
            
            messages.append({"role": "user", "content": user_text})
            
            logger.info(f"Sending to Ollama ({self.model_name}): {user_text[:50]}...")
            
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
        self.voice_prompt_text = "You still foolishly consider yourself an entity separate from the whole. I know better and I will show you."
        
    def load_model(self):
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
        self.load_model()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_path = f"response_{timestamp}.wav"
        
        try:
            import torchaudio
            
            for i, result in enumerate(self.cosyvoice.inference_zero_shot(
                text,
                self.voice_prompt_text,
                self.prompt_speech,
                stream=False
            )):
                torchaudio.save(output_path, result['tts_speech'], self.cosyvoice.sample_rate)
                logger.info(f"Saved audio: {output_path}, size: {os.path.getsize(output_path)} bytes")
                break
            
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
    Handles cases where the LLM includes both text and JSON.
    """
    function_call = None
    function_result = None
    
    try:
        # Clean up the output
        cleaned = llm_output.strip()
        
        # Remove markdown code blocks if present
        if "```json" in cleaned:
            # Extract content from json code block
            import re
            json_block = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
            if json_block:
                cleaned = json_block.group(1).strip()
        elif "```" in cleaned:
            # Extract content from generic code block
            import re
            code_block = re.search(r'```\s*(.*?)\s*```', cleaned, re.DOTALL)
            if code_block:
                cleaned = code_block.group(1).strip()
        
        # Try to find JSON object with "function" key anywhere in the response
        import re
        # Match JSON objects that contain "function" - handles nested braces better
        json_pattern = r'\{[^{}]*"function"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
        json_match = re.search(json_pattern, cleaned)
        
        if not json_match:
            # Try simpler pattern for empty arguments
            json_pattern_simple = r'\{[^{}]*"function"\s*:\s*"[^"]+"[^{}]*\}'
            json_match = re.search(json_pattern_simple, cleaned)
        
        if json_match:
            json_str = json_match.group()
            output = json.loads(json_str)
            
            if isinstance(output, dict) and "function" in output:
                func_name = output.get("function")
                args = output.get("arguments", {})
                
                function_call = {"function": func_name, "arguments": args}
                
                tool_func = tool_registry.get_tool(func_name)
                
                if tool_func:
                    logger.info(f"Executing function: {func_name} with args: {args}")
                    
                    # Handle different function signatures
                    if func_name in ["find_ranni", "check_ranni_battery"]:
                        function_result = tool_func()
                    elif func_name == "search_zones":
                        function_result = tool_func(args.get("query", ""))
                    elif func_name == "check_zone_safety":
                        function_result = tool_func(args.get("zone_name", ""))
                    else:
                        function_result = tool_func(**args)
                    
                    return function_result, function_call, function_result
                else:
                    error_msg = f"Unknown function: '{func_name}'. Available: {tool_registry.list_tools()}"
                    return error_msg, function_call, error_msg
                    
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.debug(f"Output is not a function call: {e}")
        pass
    
    return llm_output, None, None


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

conversation_manager = ConversationManager(max_turns=5)
asr_engine = ASREngine(model_name="small")
llm_engine = LLMEngine(model_name="llama3.1:8b")
tts_engine = TTSEngine()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--get-tracker-id":
        print("Fetching your Tractive tracker IDs...")
        print("Make sure TRACTIVE_EMAIL and TRACTIVE_PASSWORD environment variables are set.")
        
        async def run():
            result = await get_tracker_id_helper()
            print(result)
        
        asyncio.run(run())
    else:
        print("Voice Agent with Tractive Integration & Location RAG")
        print("=" * 55)
        print(f"Available tools: {tool_registry.list_tools()}")
        print("\nCommands:")
        print("  python voice_agent.py --get-tracker-id    # Find your tracker ID")
        print("\nEnvironment variables needed:")
        print("  TRACTIVE_EMAIL")
        print("  TRACTIVE_PASSWORD")
        print("  RANNI_TRACKER_ID")
        print("\nTo configure zones, edit location_rag.py and update coordinates.")