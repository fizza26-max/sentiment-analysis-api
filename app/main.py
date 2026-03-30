"""
FastAPI application factory.

Lifespan loads both HuggingFace models once at startup — all requests
after that are in-memory inference with no external calls.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.services.model_service import model_service
from app.api.routes import analysis, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Rate limiter (uses slowapi — thin wrapper around limits library)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])


# ---------------------------------------------------------------------------
# Lifespan — model loading happens here, not at import time
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== SentimentAPI starting up ===")
    model_service.load()          # blocks until both models are ready
    logger.info("=== Models ready — accepting requests ===")
    yield
    logger.info("=== SentimentAPI shutting down ===")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — open for portfolio demo purposes
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Routers
    app.include_router(health.router)
    app.include_router(analysis.router)

    # ---------------------------------------------------------------------------
    # Root — interactive demo UI (no frontend framework needed)
    # ---------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SentimentAPI — Live Demo</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f111a;color:#e2e8f0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 16px}
  h1{font-size:2rem;font-weight:700;background:linear-gradient(135deg,#6ee7f7,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px}
  .sub{color:#94a3b8;font-size:.95rem;margin-bottom:36px;text-align:center}
  .card{background:#1e2130;border:1px solid #2d3148;border-radius:16px;padding:28px;width:100%;max-width:640px;margin-bottom:20px}
  label{display:block;font-size:.8rem;font-weight:600;letter-spacing:.06em;color:#94a3b8;text-transform:uppercase;margin-bottom:8px}
  textarea{width:100%;background:#13151f;border:1px solid #2d3148;border-radius:10px;color:#e2e8f0;font-size:.95rem;padding:14px;resize:vertical;min-height:100px;outline:none;transition:border-color .2s}
  textarea:focus{border-color:#6ee7f7}
  button{margin-top:14px;width:100%;padding:13px;border:none;border-radius:10px;background:linear-gradient(135deg,#6ee7f7,#a78bfa);color:#0f111a;font-weight:700;font-size:1rem;cursor:pointer;transition:opacity .2s}
  button:hover{opacity:.85}
  button:disabled{opacity:.4;cursor:not-allowed}
  .result{margin-top:20px;display:none}
  .pill{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.8rem;font-weight:600;margin-right:6px;margin-bottom:6px}
  .positive{background:#064e3b;color:#6ee7b7}
  .negative{background:#4c0519;color:#fca5a5}
  .neutral{background:#1e2a40;color:#93c5fd}
  .joy{background:#3b2700;color:#fcd34d}
  .sadness{background:#1e2a40;color:#93c5fd}
  .anger{background:#4c0519;color:#fca5a5}
  .fear{background:#2e1065;color:#c4b5fd}
  .surprise{background:#0c2a1e;color:#6ee7b7}
  .disgust{background:#1a2e00;color:#a3e635}
  .bar-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .bar-label{font-size:.8rem;color:#94a3b8;min-width:80px;text-align:right}
  .bar-bg{flex:1;background:#2d3148;border-radius:4px;height:8px;overflow:hidden}
  .bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#6ee7f7,#a78bfa);transition:width .5s ease}
  .bar-pct{font-size:.75rem;color:#64748b;min-width:42px}
  .section-title{font-size:.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin:18px 0 10px}
  .links{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}
  .links a{color:#6ee7f7;font-size:.85rem;text-decoration:none;padding:6px 14px;border:1px solid #2d3148;border-radius:8px;transition:border-color .2s}
  .links a:hover{border-color:#6ee7f7}
  .spinner{display:inline-block;width:18px;height:18px;border:2px solid #2d3148;border-top-color:#6ee7f7;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:8px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .err{color:#fca5a5;margin-top:12px;font-size:.9rem}
</style>
</head>
<body>
<h1>SentimentAPI</h1>
<p class="sub">Open-source sentiment &amp; emotion analysis &bull; Powered by HuggingFace Transformers &bull; No API key needed</p>
<div class="card">
  <label for="txt">Enter text to analyse</label>
  <textarea id="txt" placeholder="Try: I just got promoted and I couldn't be happier!">I just got promoted at work and I couldn't be happier!</textarea>
  <button id="btn" onclick="run()">Analyse</button>
  <p class="err" id="err"></p>
  <div class="result" id="result">
    <div class="section-title">Sentiment</div>
    <div id="sent-pills"></div>
    <div id="sent-bars"></div>
    <div class="section-title">Emotions</div>
    <div id="emo-pills"></div>
    <div id="emo-bars"></div>
  </div>
</div>
<div class="links">
  <a href="/docs">Swagger UI</a>
  <a href="/redoc">ReDoc</a>
  <a href="/analyse/demo">Demo endpoint</a>
  <a href="/health">Health</a>
  <a href="/models">Model info</a>
</div>
<script>
async function run(){
  const txt=document.getElementById('txt').value.trim();
  const btn=document.getElementById('btn');
  const errEl=document.getElementById('err');
  errEl.textContent='';
  if(!txt)return;
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Analysing…';
  try{
    const r=await fetch('/analyse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:txt})});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||r.statusText);}
    const d=await r.json();
    render(d);
  }catch(e){
    errEl.textContent='Error: '+e.message;
  }finally{
    btn.disabled=false;btn.textContent='Analyse';
  }
}
function pill(label,cls){return`<span class="pill ${cls}">${label}</span>`;}
function bar(label,score){
  const pct=Math.round(score*100);
  return`<div class="bar-row"><span class="bar-label">${label}</span><div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div><span class="bar-pct">${pct}%</span></div>`;
}
function render(d){
  const res=document.getElementById('result');
  res.style.display='block';
  const sl=d.sentiment.label;
  document.getElementById('sent-pills').innerHTML=pill(sl.toUpperCase()+' '+Math.round(d.sentiment.score*100)+'%',sl);
  document.getElementById('sent-bars').innerHTML=d.sentiment.all_scores.map(s=>bar(s.label,s.score)).join('');
  const el=d.emotions.dominant_emotion;
  document.getElementById('emo-pills').innerHTML=pill(el.toUpperCase()+' '+Math.round(d.emotions.score*100)+'%',el);
  document.getElementById('emo-bars').innerHTML=d.emotions.all_emotions.map(e=>bar(e.label,e.score)).join('');
}
document.getElementById('txt').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')run();});
</script>
</body>
</html>"""
        return HTMLResponse(html)

    return app


app = create_app()
