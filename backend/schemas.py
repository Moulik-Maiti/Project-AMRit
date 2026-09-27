from pydantic import BaseModel, Field
from typing import Literal, Optional

# ~10 MB of sequence text; larger uploads should go through a batch pipeline
MAX_SEQUENCE_UPLOAD_CHARS = 10_000_000


class SequenceUploadRequest(BaseModel):
    sequence_data: str = Field(..., min_length=1, max_length=MAX_SEQUENCE_UPLOAD_CHARS)
    format: Optional[Literal["FASTA", "FASTQ", "CSV", "RAW"]] = "FASTA"
