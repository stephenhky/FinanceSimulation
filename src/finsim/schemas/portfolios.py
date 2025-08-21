
from datetime import date

from pydantic import BaseModel, ConfigDict


class PortfolioSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    content: dict[str, int | float]


class PortfolioTimePointScheme(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: date
    portfolio: dict[str, int | float]


class DynamicPortfolioSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    current_date: date
    timeseries: list[PortfolioTimePointScheme]
