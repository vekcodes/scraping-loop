"""Pydantic request/response models for the property-count checker."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    url: str = Field(..., description="Company website URL to analyze")


class CheckResponse(BaseModel):
    has_10_plus_properties: bool
    property_count: Optional[int] = None
    count_type: Literal["confirmed", "estimated"]
    source: str
    pages_checked: List[str]


class PageClassification(BaseModel):
    """Result of a single gpt-4o-mini page classification call."""

    type: Literal["individual_listings", "categories", "irrelevant"]
    count_if_any: Optional[int] = None
    child_links: List[str] = Field(default_factory=list)
