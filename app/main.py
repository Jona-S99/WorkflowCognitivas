from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.pages import router as pages_router
from app.api.routes.uploads import router as uploads_router
from app.api.routes.workflow import router as workflow_router


app = FastAPI(title="Agente EC")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(pages_router)
app.include_router(uploads_router)
app.include_router(workflow_router)
