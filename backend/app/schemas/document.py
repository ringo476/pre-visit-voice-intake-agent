from pydantic import BaseModel


class UploadedDocument(BaseModel):
    id: str
    filename: str
    mime_type: str
    text: str
    uploaded_at: str
