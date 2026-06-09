from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.api.runtime import templates


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "titulo_page": "Workflow EC - Inicio"},
    )


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    return templates.TemplateResponse(
        "config.html",
        {"request": request, "titulo_page": "Workflow EC - Configuración"},
    )


@router.get("/workflow", response_class=HTMLResponse)
async def workflow_page(request: Request):
    return templates.TemplateResponse(
        "workflow.html",
        {"request": request, "titulo_page": "Workflow EC - Flujo"},
    )
