import os
import json
from dotenv import load_dotenv
from typing import List, Optional

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Load .env variables
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GOOGLE_API_KEY")

class Sachverhaltselement(BaseModel):
    """
    Factual elements of Plaintiff and Defendant.
    """
    nr: int = Field(description="Die Nummer des Elements (Nr.).")
    element_name: str = Field(description="Kurzbeschreibung des Sachverhaltselements.")
    
    # Plaintiff
    plaintiff_pleading: str = Field(description="Vortrag des Klägers zum Sachverhaltselement.")
    plaintiff_document_passage: Optional[str] = Field(description="Die wörtliche Passage aus dem Dokument, die den Kläger-Vortrag belegt, falls vorhanden. Ansonsten None.", default=None)
    attachment_plaintiff: Optional[str] = Field(description="Die Anlagen (Anlage Kx) des Klägers zu diesem Element, falls vorhanden. Ansonsten None.", default=None)
    
    # Defendant
    defendant_pleading: str = Field(description="Vortrag des Beklagten zum Sachverhaltselement.")
    defendant_document_passage: Optional[str] = Field(description="Die wörtliche Passage aus dem Dokument, die den Beklagten-Vortrag belegt, falls vorhanden. Ansonsten None.", default=None)
    attachment_defendant: Optional[str] = Field(description="Die Anlagen (Anlage Bx) des Beklagten zu diesem Element, falls vorhanden. Ansonsten None.", default=None)

