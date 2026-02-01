# 🐱 Ranni Tracker - Voice Agent with GPS Location RAG

A personalized voice assistant that tracks your cat using Tractive GPS, with semantic location awareness powered by RAG (Retrieval-Augmented Generation). Features a custom Wayne June voice clone for dramatic narration in the style of *Darkest Dungeon's* Ancestor.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Voice-Activated Interface**: Push-to-talk web interface with real-time audio visualization
- **GPS Pet Tracking**: Integration with Tractive GPS trackers via REST API
- **Location RAG System**: Semantic search over custom-defined zones with polygon geofencing
- **Custom Voice Clone**: Wayne June (Darkest Dungeon narrator) voice synthesis via CosyVoice
- **Local LLM**: Runs entirely on local hardware using Ollama with Llama 3.1 8B
- **Dramatic Persona**: The assistant embodies "The Ancestor," addressing you as "The Heir"

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface                            │
│                   (Push-to-Talk Button)                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Audio/WebSocket
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Server                              │
│                      (api.py)                                   │
├─────────────────────────────────────────────────────────────────┤
│  /chat/          - Voice input → Voice output                   │
│  /api/voice-query/ - Text input → Text output                   │
│  /ranni/location/  - Direct GPS lookup                          │
│  /ranni/raw/       - Raw coordinates for calibration            │
│  /zones/           - Zone CRUD operations                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
   │ Whisper │  │ Ollama  │  │ CosyVoice│  │ Tractive │
   │  (ASR)  │  │ (LLM)   │  │  (TTS)   │  │  (GPS)   │
   └─────────┘  └────┬────┘  └──────────┘  └────┬─────┘
                     │                          │
                     ▼                          ▼
              ┌─────────────┐           ┌─────────────┐
              │ Tool Router │           │ Location    │
              │             │◄─────────►│ RAG         │
              └─────────────┘           │ (ChromaDB)  │
                                        └─────────────┘
```

## 📁 Project Structure

```
Local Voice Agent/
├── api.py              # FastAPI application with all endpoints
├── voice_agent.py      # Core voice agent: ASR, LLM, TTS, tools
├── location_rag.py     # Location RAG system with polygon zones
├── static/
│   └── index.html      # Push-to-talk web interface
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) with Llama 3.1 8B model
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (for TTS)
- Tractive GPS tracker and subscription

### Installation

1. **Clone and setup environment**
   ```bash
   conda create -n voice-agent python=3.11
   conda activate voice-agent
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn python-multipart
   pip install openai-whisper
   pip install ollama
   pip install aiotractive
   pip install chromadb shapely numpy
   pip install transformers torch torchaudio
   ```

3. **Install Ollama and pull Llama 3.1**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.1:8b
   ```

4. **Setup CosyVoice** (follow their installation guide)
   - Update paths in `voice_agent.py`:
     ```python
     COSYVOICE_BASE = r'C:\path\to\CosyVoice'
     MODEL_PATH = os.path.join(COSYVOICE_BASE, 'pretrained_models', 'CosyVoice2-0.5B')
     PROMPT_AUDIO_PATH = os.path.join(COSYVOICE_BASE, 'asset', 'zero_shot_prompt.wav')
     ```

5. **Configure Tractive credentials**
   ```bash
   # Windows
   set TRACTIVE_EMAIL=your_email@example.com
   set TRACTIVE_PASSWORD=your_password
   set RANNI_TRACKER_ID=your_tracker_id
   
   # Linux/Mac
   export TRACTIVE_EMAIL=your_email@example.com
   export TRACTIVE_PASSWORD=your_password
   export RANNI_TRACKER_ID=your_tracker_id
   ```

6. **Find your tracker ID**
   ```bash
   python voice_agent.py --get-tracker-id
   ```

7. **Run the server**
   ```bash
   python api.py
   ```

8. **Open the web interface**
   ```
   http://localhost:8000
   ```

## 🗺️ Configuring Location Zones

The Location RAG system uses polygon-defined zones to provide contextual descriptions of your cat's location.

### Zone Structure

```python
LocationZone(
    id="backyard",
    name="The Rear Gardens",
    description="the verdant hunting grounds behind the estate",
    polygon_coords=[
        (-122.001, 37.000),  # (longitude, latitude) - SW corner
        (-122.001, 37.001),  # NW corner
        (-121.999, 37.001),  # NE corner
        (-121.999, 37.000),  # SE corner
        (-122.001, 37.000),  # Close the polygon
    ],
    zone_type="yard",
    safety_level="safe",  # safe, caution, danger
    typical_activities=[
        "stalking unsuspecting birds",
        "rolling in patches of sunlight",
    ],
    tags=["outdoor", "safe", "hunting"],
    time_context={
        "morning": "Dew glistens on the grass.",
        "night": "Prime hunting conditions.",
    }
)
```

### Calibrating Zone Coordinates

1. **Start the server** and attach the tracker to yourself (or your cat)

2. **Walk to each corner** of the zone you want to define

3. **Get raw coordinates** at each point:
   ```bash
   curl http://localhost:8000/ranni/raw/
   ```
   
   Response:
   ```json
   {
     "latitude": 37.12345,
     "longitude": -122.12345,
     "polygon_format": "(-122.12345, 37.12345),  # (lon, lat)"
   }
   ```

4. **Edit `location_rag.py`** and update the `create_example_zones()` function with your coordinates

5. **Delete the database** to reload zones:
   ```bash
   rm -rf ranni_location_db/
   ```

6. **Restart the server**

### Zone Management API

```bash
# List all zones
GET /zones/

# Create a new zone
POST /zones/
{
  "id": "shed",
  "name": "The Garden Shed",
  "description": "a weathered outbuilding",
  "polygon_coords": [[-122.001, 37.001], ...],
  "zone_type": "outdoor",
  "safety_level": "safe"
}

# Update a zone
PUT /zones/{zone_id}

# Delete a zone
DELETE /zones/{zone_id}

# Semantic search zones
POST /zones/search/
{"query": "safe place for hunting"}

# Query specific coordinates
POST /zones/query/
{"longitude": -122.001, "latitude": 37.001}
```

## 🎤 Voice Commands

The assistant responds to natural language queries about your cat:

| Query | Action |
|-------|--------|
| "Where is Ranni?" | Retrieves GPS location with zone context |
| "Find my cat" | Same as above |
| "How's the tracker battery?" | Reports battery level |
| "Where does Ranni like to hunt?" | Semantic search for hunting zones |
| "Is the street safe?" | Checks zone safety level |
| "What dangerous areas are there?" | Lists all danger zones |

For general conversation, the assistant responds in character as The Ancestor with dramatic flair.

## 🔧 API Reference

### Voice Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/` | POST | Audio in → Audio out (full pipeline) |
| `/api/voice-query/` | POST | Text in → Text out (JSON body) |
| `/api/text-query/` | POST | Text in → Text out (Pydantic) |

### Ranni Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ranni/location/` | GET | Location with narrative |
| `/ranni/raw/` | GET | Raw coordinates for calibration |
| `/ranni/battery/` | GET | Battery status |

### Zone Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/zones/` | GET | List all zones |
| `/zones/` | POST | Create new zone |
| `/zones/{id}` | GET | Get specific zone |
| `/zones/{id}` | PUT | Update zone |
| `/zones/{id}` | DELETE | Delete zone |
| `/zones/query/` | POST | Query coordinates |
| `/zones/search/` | POST | Semantic search |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/` | GET | API information |
| `/health/` | GET | Health check |
| `/logs/` | GET | Interaction logs |
| `/clear/` | POST | Clear conversation |
| `/tools/` | GET | List available tools |

## 🛠️ Customization

### Changing the Voice

Edit `voice_agent.py` to use a different voice prompt:

```python
class TTSEngine:
    def __init__(self):
        # Change this to your voice prompt audio
        self.voice_prompt_text = "Your prompt text here"
        
# Update PROMPT_AUDIO_PATH to point to your reference audio
PROMPT_AUDIO_PATH = os.path.join(COSYVOICE_BASE, 'asset', 'your_voice.wav')
```

### Changing the Persona

Modify the system prompt in `LLMEngine._get_system_prompt()`:

```python
return f"""You are a helpful assistant named Jarvis.
You speak in a professional, butler-like manner.
...
"""
```

### Adding New Tools

1. Create the function in `voice_agent.py`:
   ```python
   def my_new_tool(param: str) -> str:
       return f"Result for {param}"
   ```

2. Register it:
   ```python
   tool_registry.register(
       name="my_tool",
       func=my_new_tool,
       description="Does something useful",
       parameters={"param": "string - description"}
   )
   ```

3. Update `route_llm_output()` to handle the new function signature

4. Update the system prompt to include usage examples

### Using a Different LLM

Change the model in `voice_agent.py`:

```python
llm_engine = LLMEngine(model_name="mistral:7b")  # Or any Ollama model
```

## 🐛 Troubleshooting

### "Cannot run the event loop while another loop is running"
This is handled by the `_run_async()` helper which runs async code in a thread pool.

### "torch.is_autocast_enabled() takes no arguments"
Version mismatch between transformers and PyTorch:
```bash
pip install transformers==4.40.0
```

### ChromaDB deprecated settings error
Use `PersistentClient` instead of the old `Client` with Settings (already fixed in code).

### Whisper FP16 warning
Normal on CPU - Whisper falls back to FP32 automatically.

### "Tractive email not configured"
Set environment variables before running:
```bash
set TRACTIVE_EMAIL=your@email.com
```

### Slow TTS generation
CosyVoice on CPU is slow (~5x realtime). Consider:
- Using GPU if available
- Shorter response texts
- Alternative TTS engines

## 📝 Example Interaction

```
You: "Where is Ranni?"

The Ancestor: "The Heir seeks knowledge of Ranni's whereabouts. 
Ranni dwells within The Rear Gardens - the verdant hunting grounds 
behind the estate, rich with prey and shadow. She is likely stalking 
unsuspecting birds. The tracker holds 73% of its vital essence. 
Coordinates: 37.123456, -122.123456"
```

## 🙏 Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Zero-shot voice cloning
- [Ollama](https://ollama.ai/) - Local LLM inference
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Tractive](https://tractive.com/) - GPS pet tracking
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Shapely](https://shapely.readthedocs.io/) - Geometric operations
- Wayne June - The iconic Darkest Dungeon narrator voice

## 📄 License

MIT License - See LICENSE file for details.

---

*"A trifling victory, but a victory nonetheless."* - The Ancestor
