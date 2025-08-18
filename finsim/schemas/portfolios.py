
from pydantic import BaseModel, ConfigDict


class PortfolioSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    content: dict[str, int | float]


