from utils import load_env
from components.media import PDFLoader

load_env()

pdf_loader = PDFLoader()
documents = pdf_loader.load_and_summarize(
    "/workspaces/FoodRAG/assets/healthy_nutrition/10 healthier food options at hawker centres _ The Straits Times.pdf")
print(documents)
