from fastapi import FastAPI

from .views import api_router

app = FastAPI()
app.include_router(api_router)
