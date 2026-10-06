from fastapi import FastAPI, HTTPException

from app.config import settings
from app.pipeline import support_pipeline
from app.schemas import TicketRequest, TicketResolutionResponse


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Intelligent Support Ticket Resolution Assistant "
        "using hybrid retrieval, RAG, validation, and escalation."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "application": settings.APP_NAME,
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post(
    "/tickets/resolve",
    response_model=TicketResolutionResponse,
)
def resolve_ticket(request: TicketRequest):

    try:

        result = support_pipeline.resolve(
            query=request.ticket,
            retrieval_limit=5,
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status": "escalate",
                "reason": "internal_error",
                "message": str(exc),
            },
        )