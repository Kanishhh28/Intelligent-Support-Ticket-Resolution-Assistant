from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    ticket: str = Field(
        ...,
        min_length=1,
        description="Customer support ticket text.",
    )


class Source(BaseModel):
    citation: str
    source_id: str
    title: str | None = None
    rrf_score: float | None = None


class ValidationResult(BaseModel):
    status: str
    valid: bool
    reason: str
    validation_score: float
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    missing_steps: list[int] = Field(default_factory=list)


class QueryAnalysis(BaseModel):
    intent: str
    product: str
    severity: str
    sentiment: str
    normalized_query: str = ""
    confidence: float = 0.0
    provider: str = "unknown"


class TicketResolutionResponse(BaseModel):
    status: str
    reason: str
    answer: str
    analysis: QueryAnalysis | None = None
    sources: list[Source] = Field(default_factory=list)
    validation: ValidationResult | None = None