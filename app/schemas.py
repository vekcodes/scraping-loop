"""Pydantic request/response models for the property-count checker."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    url: str = Field(..., description="Company website URL to analyze")


# How a page's count was derived — surfaced so an "exact" number is auditable.
CountBasis = Literal[
    "stated_total",   # page explicitly says e.g. "12 properties" / "of 84"
    "pagination",     # inferred from "Page 1 of N" / "Showing X of Y"
    "counted_items",  # counted distinct listing tiles / detail links
    "unknown",
]


class PageCount(BaseModel):
    """Per-page contribution to the total, for auditing the final count."""

    url: str
    count: Optional[int] = None
    count_basis: CountBasis = "unknown"


class CheckResponse(BaseModel):
    has_10_plus_properties: bool
    property_count: Optional[int] = None
    count_type: Literal["confirmed", "estimated"]
    source: str
    pages_checked: List[str]
    # Per-listing-page breakdown of where the total came from.
    breakdown: List[PageCount] = Field(default_factory=list)


class PageClassification(BaseModel):
    """Result of a single page classification call."""

    type: Literal["individual_listings", "categories", "irrelevant"]
    count_if_any: Optional[int] = None
    count_basis: CountBasis = "unknown"
    child_links: List[str] = Field(default_factory=list)
