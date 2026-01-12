from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "rf": BASE_DIR / "models/rf_model.pkl",
    "dt": BASE_DIR / "models/dt_model.pkl",
    "lgr": BASE_DIR / "models/lgr_model.pkl",
    "gb": BASE_DIR / "models/gb_model.pkl",
}
