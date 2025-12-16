"""
FastAPI Application for Voice Agent with Function Calling
==========================================================
Personalized setup:
- Wayne June voice clone
- Local Ollama with Llama 3.1 8B
- Function calling for arXiv search and calculations
"""

import os
import logging
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import core voice agent components
from voice_agent import (
    tool_registry,
    conversation_manager,
    asr_engine,
    llm_engine,
    tts_engine,
    route_llm_output,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Voice Agent with Function Calling",
    description="Wayne June voiced assistant with arXiv search and calculator",
    version="2.0.0"
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TextQueryRequest(BaseModel):
    """Request model for text-based queries."""
    text: str


class QueryResponse(BaseModel):
    """Response model for queries."""
    response: str
    function_call: Optional[Dict] = None
    function_result: Optional[str] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Voice Agent with Function Calling (Wayne June Edition)",
        "version": "2.0.0",
        "llm": "Llama 3.1 8B (Ollama)",
        "voice": "Wayne June clone (CosyVoice)",
        "endpoints": {
            "/chat/": "POST - Audio input, audio output",
            "/api/voice-query/": "POST - Text input, text output",
            "/api/text-query/": "POST - Text input (Pydantic)",
            "/logs/": "GET - Get interaction logs",
            "/clear/": "POST - Clear conversation history",
            "/tools/": "GET - List available tools"
        },
        "available_tools": tool_registry.list_tools()
    }


@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    """
    Main voice chat endpoint with function calling.
    
    Pipeline: Audio -> ASR -> LLM (with tools) -> TTS -> Audio
    """
    try:
        # Step 1: Receive and transcribe audio
        audio_bytes = await file.read()
        user_text = asr_engine.transcribe(audio_bytes)
        print(f"User said: {user_text}")
        
        # Add to conversation history
        conversation_manager.add_user_message(user_text)
        
        # Step 2: Generate response with LLM (may include function call)
        history = conversation_manager.get_ollama_history()
        llm_response = llm_engine.generate_response(user_text, history[:-1])  # Exclude current message
        print(f"LLM raw response: {llm_response}")
        
        # Step 3: Route response - execute function if needed
        final_response, function_call, function_result = route_llm_output(llm_response)
        print(f"Final response: {final_response}")
        
        # Log the interaction
        conversation_manager.log_interaction(
            user_query=user_text,
            llm_response=llm_response,
            function_call=function_call,
            function_result=function_result,
            final_response=final_response
        )
        
        # Add response to history
        conversation_manager.add_assistant_message(final_response)
        
        # Step 4: Convert response to speech (Wayne June voice)
        audio_path = tts_engine.synthesize(final_response)
        print(f"Audio generated: {audio_path}")
        
        # Step 5: Return audio response
        return FileResponse(audio_path, media_type="audio/wav")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Chat endpoint error: {e}")
        return {"error": str(e)}


@app.post("/api/voice-query/", response_model=QueryResponse)
async def voice_query_endpoint(request: dict):
    """
    Text-based query endpoint with function calling.
    Useful for testing without audio.
    """
    user_text = request.get("text", "")
    
    if not user_text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    
    # Add to conversation history
    conversation_manager.add_user_message(user_text)
    
    # Generate LLM response
    history = conversation_manager.get_ollama_history()
    llm_response = llm_engine.generate_response(user_text, history[:-1])
    
    # Route response
    final_response, function_call, function_result = route_llm_output(llm_response)
    
    # Log interaction
    conversation_manager.log_interaction(
        user_query=user_text,
        llm_response=llm_response,
        function_call=function_call,
        function_result=function_result,
        final_response=final_response
    )
    
    # Add to history
    conversation_manager.add_assistant_message(final_response)
    
    return QueryResponse(
        response=final_response,
        function_call=function_call,
        function_result=function_result
    )


@app.post("/api/text-query/", response_model=QueryResponse)
async def text_query_endpoint(request: TextQueryRequest):
    """Text query endpoint with Pydantic model validation."""
    return await voice_query_endpoint({"text": request.text})


@app.get("/logs/")
async def get_logs():
    """Get all interaction logs for debugging/deliverables."""
    return {
        "logs": conversation_manager.get_logs(),
        "conversation_history": conversation_manager.get_history()
    }


@app.post("/clear/")
async def clear_history():
    """Clear conversation history."""
    conversation_manager.clear()
    return {"message": "Conversation history cleared"}


@app.get("/tools/")
async def list_tools():
    """List available tools and their descriptions."""
    tools = []
    for name in tool_registry.list_tools():
        tool = tool_registry.tools[name]
        tools.append({
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"]
        })
    return {"tools": tools}


@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "asr_model": asr_engine.model_name,
        "llm_model": llm_engine.model_name,
        "tts_voice": "Wayne June (CosyVoice)",
        "conversation_turns": len(conversation_manager.history) // 2,
        "tools_available": tool_registry.list_tools()
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    # Run without threading - use dedicated terminal as you prefer
    uvicorn.run(app, host="0.0.0.0", port=8000)
