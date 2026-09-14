from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SingleIsolateInput(BaseModel):
    species: str
    target_antibiotics: List[str]
    mutations: List[str]
    patient_age: Optional[int] = 35
    patient_gender: Optional[str] = "M"
    infection_site: Optional[str] = "Blood"
    clinical_severity: Optional[str] = "Moderate"

class SequenceUploadRequest(BaseModel):
    sequence_data: str
    format: Optional[str] = "FASTA"
