from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from uat_suggestor import uat_manager

router = APIRouter(prefix="/uatkeywordgen", tags=["uatkeywordgen"])

class InData(BaseModel):
    data: Dict[str, str]

class UATRequest(BaseModel):
    indata: InData

@router.post("/suggest", response_model=Dict[str, Any])
async def suggest_uat(request: UATRequest):
    try:
        abstract =request.indata.data.get("abstract")
        print(abstract)
        if not abstract:
            raise ValueError("Abstract is missing from the input data")
        
        result = uat_manager(abstract)
        print(result)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "healthy"}