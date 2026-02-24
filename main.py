import os
import uuid
import gc

# -------------------------------
# 🔧 Paddle memory stability flags
# -------------------------------
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"   # limit CPU threads
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil

from paddlex import create_pipeline

# -------------------------------
# 🚀 FastAPI App
# -------------------------------
app = FastAPI(title="PaddleX Table Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 🧠 Load pipeline ONCE (important)
# -------------------------------
print("Loading PaddleX pipeline...")

pipeline = create_pipeline(
    "table_recognition_v2",
    device="cpu"
)

print("Pipeline ready 🚀")


# -------------------------------
# ❤️ Health check
# -------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


# -------------------------------
# 📄 Table Recognition Endpoint
# -------------------------------
@app.post("/table-recognition")
async def table_recognition(file: UploadFile = File(...)):
    temp_path = f"/tmp/{uuid.uuid4()}.jpg"

    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # IMPORTANT: don't convert to list()
        results = []
        for res in pipeline.predict(temp_path):
            if hasattr(res, "to_dict"):
                results.append(res.to_dict())
            elif hasattr(res, "__dict__"):
                results.append(res.__dict__)
            else:
                results.append(str(res))

        return JSONResponse(content={"result": results})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Force memory cleanup
        gc.collect()


# -------------------------------
# ▶ Run manually
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1   # IMPORTANT: do NOT increase workers on low RAM
    )
