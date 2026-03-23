"""
MAIN — Entry Point
-------------------
Run with:
    uvicorn main:app --reload --port 8000

Swagger docs available at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from models.trend_prediction_engine import TrendPredictionEngine

# ------------------------------------------------------------------ #
#  App setup                                                          #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Meme Coin Trend Prediction API",
    description=(
        "Multi-engine social signal analyzer that predicts meme coin trends "
        "using Reddit, Twitter, and Discord data."
    ),
    version="1.0.0",
)

# Allow frontend (dashboard) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  Register routes                                                    #
# ------------------------------------------------------------------ #

app.include_router(router, prefix="/api/v1")


# ------------------------------------------------------------------ #
#  Startup: auto-train model if no saved model found                  #
# ------------------------------------------------------------------ #

@app.on_event("startup")
async def startup_event():
    import os
    model_path = "models/trend_model.pkl"
    predictor  = TrendPredictionEngine(use_ml=True)

    if os.path.exists(model_path):
        print(f"[Startup] Loading saved model from {model_path}")
        predictor.load_model(model_path)
    else:
        print("[Startup] No saved model found — training on synthetic data...")
        predictor.train()

    # Store on app state so routes can reuse
    app.state.predictor = predictor
    print("[Startup] API is ready.")


# ------------------------------------------------------------------ #
#  Root                                                               #
# ------------------------------------------------------------------ #

@app.get("/")
def root():
    return {
        "message": "Meme Coin Trend Prediction API",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }
