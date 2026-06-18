from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import Optional, Union, Any

# ==========================================
# 1. METRICS & KPI SCHEMAS
# ==========================================
class MetricNode(BaseModel):
    """Schema for individual quantitative metrics extracted from text."""
    label: str = Field(default="Extracted Metric", validation_alias=AliasChoices('label', 'name', 'title', 'metric'))
    value: Union[str, float] = Field(default="N/A")
    unit: Optional[str] = Field(default="")
    trend: Optional[str] = Field(default="")
    source_citation: str = Field(default="Source not provided in text", validation_alias=AliasChoices('source_citation', 'source', 'citation'))
    
    model_config = ConfigDict(extra="ignore")

# ==========================================
# 2. DYNAMIC ALTAIR CHART SCHEMAS (NEW)
# ==========================================
class ChartConfig(BaseModel):
    """Schema for the LLM to dynamically reason about and configure an Altair chart."""
    chart_title: str = Field(default="Data Visualization")
    reasoning: str = Field(
        default="This chart highlights the primary trends in the dataset.", 
        description="A brief explanation of WHY the LLM chose this specific chart type and data pairing."
    )
    chart_type: str = Field(default="bar", description="Must be exactly: 'bar', 'line', or 'scatter'")
    x_axis: str = Field(default="", description="The exact column name from the dataset to map to the X-axis.")
    y_axes: list[str] = Field(default_factory=list, description="List of column names from the dataset to map to the Y-axis. The backend will melt these.")
    dataset: list[dict[str, Any]] = Field(
        default_factory=list, 
        description="The raw tabular data extracted specifically for this chart."
    )

    model_config = ConfigDict(extra="ignore")

class MultiChartDashboard(BaseModel):
    """Master schema for the Semantic Reasoning Dashboard and One-Click Exports."""
    dashboard_title: str = Field(default="Executive Analytical Dashboard")
    executive_summary: str = Field(default="Synthesis of quantitative data and tabular insights.")
    metrics: list[MetricNode] = Field(default_factory=list)
    charts: list[ChartConfig] = Field(
        default_factory=list, 
        description="The dynamically reasoned chart configurations."
    )
    
    model_config = ConfigDict(extra="ignore")

# ==========================================
# 3. NETWORK GRAPH SCHEMAS
# ==========================================
class GraphEdge(BaseModel):
    """Schema for defining connections between entities in the cross-document graph."""
    source: str = Field(default="Node A", validation_alias=AliasChoices('source', 'from', 'origin'))
    target: str = Field(default="Node B", validation_alias=AliasChoices('target', 'to', 'destination'))
    relationship: str = Field(default="Connects to") 
    
    model_config = ConfigDict(extra="ignore")

class GraphData(BaseModel):
    """Master schema for synthesizing the entity relationship graph."""
    edges: list[GraphEdge] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="ignore")