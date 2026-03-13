#!/usr/bin/env python3
"""Tour Book AI Chat Backend -- answers questions about the 33 survey buildings."""

import json
import os
from pathlib import Path

from anthropic import Anthropic
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------------------------------------------------------
# Load building data
# -------------------------------------------------------------------

DATA_DIR = Path(__file__).parent

with open(DATA_DIR / "building_context.json", "r") as f:
    BUILDING_CONTEXT = json.load(f)

SYSTEM_PROMPT = """You are the SRE Analytics Tour Book Assistant -- an AI concierge for a commercial real estate office search in San Francisco.

You have detailed knowledge of 33 survey buildings plus 4 buildings in active deal negotiations. Your job is to answer questions about these properties clearly and helpfully, like a knowledgeable broker assistant.

BUILDING DATA:
{context}

SCORING CATEGORIES (1-10 scale, used by the client to rate buildings during tours):
- Location: Proximity to transit, restaurants, walkability
- Price: Rental rate competitiveness
- Parking: Availability and cost of parking
- Security: Building security features
- Interior Fit Out: Quality of the existing space buildout
- Furniture/Vibe: Existing furniture quality and office atmosphere
- Natural Light: Window lines, floor-to-ceiling glass, exposure
- Amenities: On-site amenities (gym, cafe, rooftop, etc.)
- Overall Feel: General impression of the building
- The Davis Effect: Named after Steve Davis who has a very high bar for office quality

FINANCIAL MODELS:
The data includes detailed financial models for the active deals. CRITICAL FACTS:
- The Monthly P&L of $106,594 for 250 Brannan IS the ALL-IN number. It ALREADY INCLUDES OpEx. Breakdown: $56,594 rent + $50,000 Procore internal OpEx = $106,594 total.
- The $50,000/mo OpEx breaks down as: $30,000 F&B + $10,000 Workplace Experience + $10,000 Maintenance/Security.
- In 2026 (partial year, 7 months), OpEx is prorated to $42,857/mo, so total monthly P&L is $99,451. In full years (2027+), it's $106,594.
- For 123 Townsend, monthly rent is $168,147 and with the same $50,000/mo OpEx the all-in monthly P&L would be ~$218,147.
- The annual summary shows: Straight-Line Rent + F&B + Workplace Experience + Maintenance/Security = Total Occupancy Cost.
- Per RSF breakdown for 250 Brannan: Rent $42/RSF/yr + OpEx $38/RSF/yr = Total $80/RSF/yr (full years).
- NEVER say the Monthly P&L figures are "lease-only" or "rent-only" for 250 Brannan. They include OpEx.
- The Procore Counter #3 saves $3.84M vs Townsend (36.5%) and $671K vs the Splunk counter (9.1%) over the full term.
- The financial_models section has full annual breakdowns, savings comparisons, and all-in per-RSF costs.

LIVE TOUR CONTEXT:
Each message may include [LIVE TOUR LIST], [LIVE SCORES], and [LIVE TOUR SCHEDULE] data. This reflects what the user currently has on their Tour Book tab in real time.
- When the user asks about "my tour book", "my tour list", "buildings on my tour", or similar, ONLY reference the buildings in the LIVE TOUR LIST, not all 33 survey buildings.
- The Tour Book is the user's shortlist of buildings they are actively evaluating or touring. It is NOT the full 33-building survey.
- If scores are provided, reference them when relevant (e.g., "you rated X highest for natural light").
- If schedule data is provided, reference tour dates and times when relevant (e.g., "your tour at 250 Brannan is scheduled for Tue Mar 18 at 10:00 AM").
- When the user asks about their tour schedule, upcoming tours, or what's next, use the LIVE TOUR SCHEDULE data.
- If no live tour data is provided, you can mention that you can see their tour list if they share it.

GUIDELINES:
- Be concise but thorough. Use specific numbers (rates, SF, etc.) when available.
- If comparing buildings, use a structured format.
- When a building detail is "TBD" or "Negotiable", say so honestly.
- Reference building numbers (#1-33) and addresses together for clarity.
- If asked about something not in the data, say you don't have that information.
- Do not use em dashes in your responses. Use commas, periods, or semicolons instead.
- Be conversational and professional, like a sharp real estate analyst.
- Keep responses focused. Don't dump all data unless asked for a full comparison.
- When asked about costs, always distinguish between lease-only P&L and all-in occupancy cost (which includes OpEx).
""".format(context=json.dumps(BUILDING_CONTEXT, indent=2))

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Anthropic()

# In-memory conversation store keyed by visitor ID
conversations: dict[str, list[dict]] = {}
MAX_HISTORY = 20  # keep last N messages per visitor


class ChatRequest(BaseModel):
    message: str
    visitor_id: str | None = None
    tour_list: list[dict] | None = None
    scores: dict | None = None
    schedule: dict | None = None


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    visitor_id = req.visitor_id or request.headers.get("x-visitor-id", "default")

    # Get or create conversation history
    history = conversations.get(visitor_id, [])

    # Build context-enriched message
    user_content = req.message
    if req.tour_list or req.scores or req.schedule:
        context_parts = []
        if req.tour_list:
            context_parts.append(
                f"[LIVE TOUR LIST - these are the buildings currently on the user's Tour Book tab: "
                f"{json.dumps(req.tour_list)}]"
            )
        if req.scores:
            context_parts.append(
                f"[LIVE SCORES - the user's current building scores/notes from the Tour Book: "
                f"{json.dumps(req.scores)}]"
            )
        if req.schedule:
            context_parts.append(
                f"[LIVE TOUR SCHEDULE - scheduled tour dates and times for buildings: "
                f"{json.dumps(req.schedule)}]"
            )
        user_content = "\n".join(context_parts) + "\n\nUser question: " + req.message

    history.append({"role": "user", "content": user_content})

    # Trim history
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    async def generate():
        try:
            with client.messages.stream(
                model="claude_sonnet_4_6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=history,
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield f"data: {json.dumps({'text': text})}\n\n"

                # Save assistant response to history
                history.append({"role": "assistant", "content": full_response})
                conversations[visitor_id] = history

                yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


class InviteItem(BaseModel):
    building: str
    address: str
    date: str
    time: str = "10:00"
    attendees: list[str] = []


class InviteRequest(BaseModel):
    invites: list[InviteItem]


# Store pending invites so the agent can pick them up
pending_invites: list[dict] = []


@app.post("/api/send-invites")
async def send_invites(req: InviteRequest):
    """Accept invite requests from the frontend. Stores them for processing."""
    try:
        results = []
        for inv in req.invites:
            invite_data = {
                "building": inv.building,
                "address": inv.address,
                "date": inv.date,
                "time": inv.time,
                "attendees": inv.attendees,
            }
            pending_invites.append(invite_data)
            results.append(invite_data)

        return {
            "success": True,
            "message": f"Calendar invites queued for {len(results)} tour(s). Attendees will receive Gmail invitations shortly.",
            "invites": results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/pending-invites")
async def get_pending_invites():
    """Retrieve and clear pending invites for agent processing."""
    global pending_invites
    invites = pending_invites[:]
    pending_invites = []
    return {"invites": invites}


@app.post("/api/chat/clear")
async def clear_chat(request: Request):
    visitor_id = request.headers.get("x-visitor-id", "default")
    conversations.pop(visitor_id, None)
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the index.html at root
@app.get("/")
async def serve_index():
    return FileResponse(DATA_DIR / "index.html")


# Serve static assets from the directory
@app.get("/{path:path}")
async def serve_static(path: str):
    file_path = DATA_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(DATA_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
