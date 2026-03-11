import fitz  # PyMuPDF

# open PDF
doc = fitz.open("pdf/eu_ai_act.pdf")

text = ""

for page in doc:
    text += page.get_text()

print(text[:2000])