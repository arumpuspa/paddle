from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from paddlex import create_pipeline
from fastapi.encoders import jsonable_encoder

app = FastAPI(title="PaddleX Table Recognition API")

# -------------------------------------------------
# Load pipeline ONCE at startup (important for speed)
# -------------------------------------------------
print("Loading PaddleX pipeline...")

pipeline = create_pipeline(
    "table_recognition_v2",
    device="cpu"   # change to "gpu:0" if using GPU
)

print("Pipeline ready 🚀")


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


# -------------------------------------------------
# Table recognition endpoint
# -------------------------------------------------

@app.post("/table-recognition")
async def table_recognition(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"

        # Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # PaddleX returns generator
        results = list(pipeline.predict(temp_path))

        # Delete temp file
        os.remove(temp_path)

        # Convert to JSON-safe format
        return jsonable_encoder({"result": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# -------------------------------------------------
# Run locally (optional)
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
