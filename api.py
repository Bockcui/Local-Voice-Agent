"""
FastAPI Application for Voice Agent with Location RAG
======================================================
Personalized setup:
- Wayne June voice clone
- Local Ollama with Llama 3.1 8B
- Tractive GPS tracking for Ranni the cat
- Vector search + polygon geofencing for location context
"""

import os
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import core voice agent components
from voice_agent import (
    tool_registry,
    conversation_manager,
    asr_engine,
    llm_engine,
    tts_engine,
    route_llm_output,
    get_ranni_location,
    get_ranni_battery,
    search_ranni_zones,
    check_zone_safety,
    get_location_rag
)

from location_rag import LocationZone, LocationContext

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
    title="Voice Agent with Location RAG",
    description="Wayne June voiced assistant with semantic location search and polygon geofencing for Ranni",
    version="4.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


class ZoneCreateRequest(BaseModel):
    """Request model for creating a new zone."""
    id: str
    name: str
    description: str
    polygon_coords: List[Tuple[float, float]]  # List of (lon, lat) tuples
    zone_type: str = "general"
    safety_level: str = "safe"
    typical_activities: List[str] = []
    tags: List[str] = []
    time_context: Dict[str, str] = {}


class ZoneUpdateRequest(BaseModel):
    """Request model for updating a zone."""
    name: Optional[str] = None
    description: Optional[str] = None
    polygon_coords: Optional[List[Tuple[float, float]]] = None
    zone_type: Optional[str] = None
    safety_level: Optional[str] = None
    typical_activities: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    time_context: Optional[Dict[str, str]] = None


class LocationQueryRequest(BaseModel):
    """Request for querying a specific location."""
    longitude: float
    latitude: float
    semantic_query: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""
    query: str
    n_results: int = 3
    filter_type: Optional[str] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    else:
        return HTMLResponse(content="<h1>Frontend not found. Access /api/ for API info.</h1>", status_code=200)


@app.get("/api/")
async def api_info():
    """API information endpoint."""
    rag = get_location_rag()
    
    return {
        "message": "Voice Agent with Location RAG (Wayne June Edition)",
        "version": "4.0.0",
        "llm": "Llama 3.1 8B (Ollama)",
        "voice": "Wayne June clone (CosyVoice)",
        "cat": "Ranni",
        "location_system": "LangChain FAISS + GeoPandas polygon geofencing",
        "zones_configured": len(rag.zones),
        "endpoints": {
            "/chat/": "POST - Audio input, audio output",
            "/api/voice-query/": "POST - Text input, text output",
            "/api/text-query/": "POST - Text input (Pydantic)",
            "/ranni/location/": "GET - Direct location lookup",
            "/ranni/battery/": "GET - Tracker battery status",
            "/zones/": "GET/POST - List or create zones",
            "/zones/{zone_id}": "GET/PUT/DELETE - Manage specific zone",
            "/zones/query/": "POST - Query location context",
            "/zones/search/": "POST - Semantic search zones",
            "/logs/": "GET - Get interaction logs",
            "/clear/": "POST - Clear conversation history",
            "/tools/": "GET - List available tools"
        },
        "available_tools": tool_registry.list_tools()
    }


# ============================================================================
# VOICE CHAT ENDPOINTS
# ============================================================================

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    """
    Main voice chat endpoint with function calling.
    Pipeline: Audio -> ASR -> LLM (with tools) -> TTS -> Audio
    """
    try:
        audio_bytes = await file.read()
        user_text = asr_engine.transcribe(audio_bytes)
        print(f"User said: {user_text}")
        
        conversation_manager.add_user_message(user_text)
        
        history = conversation_manager.get_ollama_history()
        llm_response = llm_engine.generate_response(user_text, history[:-1])
        print(f"LLM raw response: {llm_response}")
        
        final_response, function_call, function_result = route_llm_output(llm_response)
        print(f"Final response: {final_response}")
        
        conversation_manager.log_interaction(
            user_query=user_text,
            llm_response=llm_response,
            function_call=function_call,
            function_result=function_result,
            final_response=final_response
        )
        
        conversation_manager.add_assistant_message(final_response)
        
        audio_path = tts_engine.synthesize(final_response)
        print(f"Audio generated: {audio_path}")
        
        return FileResponse(audio_path, media_type="audio/wav")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Chat endpoint error: {e}")
        return {"error": str(e)}


@app.post("/api/voice-query/", response_model=QueryResponse)
async def voice_query_endpoint(request: dict):
    """Text-based query endpoint with function calling."""
    user_text = request.get("text", "")
    
    if not user_text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    
    conversation_manager.add_user_message(user_text)
    
    history = conversation_manager.get_ollama_history()
    llm_response = llm_engine.generate_response(user_text, history[:-1])
    
    final_response, function_call, function_result = route_llm_output(llm_response)
    
    conversation_manager.log_interaction(
        user_query=user_text,
        llm_response=llm_response,
        function_call=function_call,
        function_result=function_result,
        final_response=final_response
    )
    
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


# ============================================================================
# RANNI DIRECT ENDPOINTS
# ============================================================================

@app.get("/ranni/location/")
async def get_ranni_location_direct():
    """Get Ranni's current location directly (bypasses LLM)."""
    try:
        location_info = get_ranni_location()
        return {"status": "success", "narrative": location_info}
    except Exception as e:
        logger.error(f"Location lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ranni/battery/")
async def get_ranni_battery_direct():
    """Get Ranni's tracker battery status directly."""
    try:
        battery_info = get_ranni_battery()
        return {"status": "success", "narrative": battery_info}
    except Exception as e:
        logger.error(f"Battery lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ZONE MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/zones/")
async def list_zones():
    """List all configured zones."""
    rag = get_location_rag()
    
    zones_list = []
    for zone_id, zone in rag.zones.items():
        zones_list.append({
            "id": zone.id,
            "name": zone.name,
            "description": zone.description,
            "zone_type": zone.zone_type,
            "safety_level": zone.safety_level,
            "polygon_coords": zone.polygon_coords,
            "typical_activities": zone.typical_activities,
            "tags": zone.tags,
            "time_context": zone.time_context
        })
    
    return {
        "total": len(zones_list),
        "zones": zones_list
    }


@app.post("/zones/")
async def create_zone(request: ZoneCreateRequest):
    """Create a new zone."""
    rag = get_location_rag()
    
    if request.id in rag.zones:
        raise HTTPException(status_code=400, detail=f"Zone with ID '{request.id}' already exists")
    
    zone = LocationZone(
        id=request.id,
        name=request.name,
        description=request.description,
        polygon_coords=request.polygon_coords,
        zone_type=request.zone_type,
        safety_level=request.safety_level,
        typical_activities=request.typical_activities,
        tags=request.tags,
        time_context=request.time_context
    )
    
    rag.add_zone(zone)
    rag.persist()
    
    return {"status": "created", "zone_id": zone.id, "name": zone.name}


@app.get("/zones/{zone_id}")
async def get_zone(zone_id: str):
    """Get a specific zone by ID."""
    rag = get_location_rag()
    
    if zone_id not in rag.zones:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")
    
    zone = rag.zones[zone_id]
    
    return {
        "id": zone.id,
        "name": zone.name,
        "description": zone.description,
        "zone_type": zone.zone_type,
        "safety_level": zone.safety_level,
        "polygon_coords": zone.polygon_coords,
        "typical_activities": zone.typical_activities,
        "tags": zone.tags,
        "time_context": zone.time_context
    }


@app.put("/zones/{zone_id}")
async def update_zone(zone_id: str, request: ZoneUpdateRequest):
    """Update an existing zone."""
    rag = get_location_rag()
    
    if zone_id not in rag.zones:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")
    
    existing = rag.zones[zone_id]
    
    # Update fields
    updated_zone = LocationZone(
        id=zone_id,
        name=request.name if request.name else existing.name,
        description=request.description if request.description else existing.description,
        polygon_coords=request.polygon_coords if request.polygon_coords else existing.polygon_coords,
        zone_type=request.zone_type if request.zone_type else existing.zone_type,
        safety_level=request.safety_level if request.safety_level else existing.safety_level,
        typical_activities=request.typical_activities if request.typical_activities else existing.typical_activities,
        tags=request.tags if request.tags else existing.tags,
        time_context=request.time_context if request.time_context else existing.time_context
    )
    
    rag.add_zone(updated_zone)  # This will upsert
    rag.persist()
    
    return {"status": "updated", "zone_id": zone_id}


@app.delete("/zones/{zone_id}")
async def delete_zone(zone_id: str):
    """Delete a zone."""
    rag = get_location_rag()
    
    if zone_id not in rag.zones:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")
    
    rag.remove_zone(zone_id)
    rag.persist()
    
    return {"status": "deleted", "zone_id": zone_id}


@app.post("/zones/query/")
async def query_location(request: LocationQueryRequest):
    """Query location context for specific coordinates."""
    rag = get_location_rag()
    
    context = rag.get_location_context(
        lon=request.longitude,
        lat=request.latitude,
        semantic_query=request.semantic_query
    )
    
    return {
        "coordinates": {"longitude": request.longitude, "latitude": request.latitude},
        "zone": {
            "id": context.zone.id if context.zone else None,
            "name": context.zone.name if context.zone else None,
            "safety_level": context.zone.safety_level if context.zone else None
        } if context.zone else None,
        "is_inside_zone": context.is_inside_zone,
        "distance_m": context.distance_m,
        "narrative": context.narrative,
        "safety_alert": context.safety_alert,
        "semantic_matches": [
            {"name": m["zone"].name, "score": m["score"]} 
            for m in context.semantic_matches
        ],
        "timestamp": context.timestamp
    }


@app.post("/zones/search/")
async def search_zones(request: SemanticSearchRequest):
    """Perform semantic search on zones."""
    rag = get_location_rag()
    
    results = rag.semantic_search(
        query=request.query,
        n_results=request.n_results,
        filter_type=request.filter_type
    )
    
    return {
        "query": request.query,
        "results": [
            {
                "id": match["zone"].id,
                "name": match["zone"].name,
                "description": match["zone"].description,
                "score": match["score"],
                "zone_type": match["zone"].zone_type,
                "safety_level": match["zone"].safety_level
            }
            for match in results
        ]
    }


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/logs/")
async def get_logs():
    """Get all interaction logs."""
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
    """List available tools."""
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
    rag = get_location_rag()
    
    return {
        "status": "healthy",
        "asr_model": asr_engine.model_name,
        "llm_model": llm_engine.model_name,
        "tts_voice": "Wayne June (CosyVoice)",
        "conversation_turns": len(conversation_manager.history) // 2,
        "tools_available": tool_registry.list_tools(),
        "zones_configured": len(rag.zones),
        "cat_name": "Ranni"
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Voice Agent with Location RAG...")
    print(f"Available tools: {tool_registry.list_tools()}")
    
    # Initialize RAG on startup
    rag = get_location_rag()
    print(f"Loaded {len(rag.zones)} location zones")
    
    print("\nMake sure to set environment variables:")
    print("  TRACTIVE_EMAIL")
    print("  TRACTIVE_PASSWORD")
    print("  RANNI_TRACKER_ID")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)