"""FastAPI backend for the Suzuka telemetry app.

Routes
  /                static frontend (frontend/)
  /bundle.json     precomputed data bundle (build/bundle.json)
  /czml/{name}     CZML documents (build/czml/)
  /ws/live         WebSocket: streams lap 2 at native 100 Hz pacing, looping

Run:  uvicorn app:app --host 127.0.0.1 --port 8321   (from server/)
"""
import asyncio
import os

import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(HERE, "frontend")
BUILD = os.path.join(HERE, "build")

app = FastAPI(title="Suzuka Telemetry & Predictive Line")


@app.get("/bundle.json")
def bundle():
    return FileResponse(os.path.join(BUILD, "bundle.json"))


@app.get("/czml/{name}")
def czml(name: str):
    safe = name if name.endswith(".czml") else name + ".czml"
    return FileResponse(os.path.join(BUILD, "czml", safe))


async def _stream_lap(ws: WebSocket) -> None:
    df = pd.read_csv(os.path.join(HERE, "outputs", "telemetry_driver.csv"))
    sub = df[df["lap"] == 2]
    while True:
        for r in sub.itertuples(index=False):
            await ws.send_json({
                "lat": round(r.lat, 7), "lon": round(r.lon, 7),
                "alt": float(r.alt_m), "speed_kmh": round(float(r.speed_ms) * 3.6, 1),
            })
            await asyncio.sleep(0.01)   # native 100 Hz pacing
        await asyncio.sleep(2.0)        # brief pause between looped laps


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    try:
        await _stream_lap(ws)
    except Exception:
        pass  # client disconnected


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="static")
