from crewai_tools import PDFSearchTool
from config import get_embedding_config


def get_pdf_tool(pdf_path: str) -> PDFSearchTool:
    return PDFSearchTool(
        pdf=pdf_path,
        config={"embedding_model": get_embedding_config()},
    )

