import os
from pathlib import Path


RAW_DATA_DIR_PATH=Path(os.getenv("RAW_DATA_DIR_PATH", "data"))
EXP_DIR_PATH=Path(os.getenv("EXP_DIR_PATH", "exp"))