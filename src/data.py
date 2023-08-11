from pydantic import BaseModel

class Data(BaseModel):
    id: str
    rows: list
    columns: list