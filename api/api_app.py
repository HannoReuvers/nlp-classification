from fastapi import FastAPI

# Import routers
from api.routes.bow.router import router as bow_router

app = FastAPI(
    title="Review Classification API",
    description="API for classifying reviews as positive or negative using a pre-trained model.",
    version="1.0.0",
)


@app.get(
    "/",
    tags=["root"],
    summary="Main Endpoint",
    description="Returns a welcome message for the Review Classification API.",
)
def read_root():
    return {"message": "Review Classification API"}


# ---------------------- ADD APP ROUTERS ----------------------#
app.include_router(bow_router)
