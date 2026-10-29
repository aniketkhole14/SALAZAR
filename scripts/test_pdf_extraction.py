from pathlib import Path

import fitz

root = Path(__file__).resolve().parents[1]
documents_dir = root / "data" / "documents"

pdf_files = sorted(documents_dir.glob("*.pdf"))

if not pdf_files:
    raise SystemExit(
        f"No PDF files found in {documents_dir}\n"
        "Copy a text-based PDF into data/documents/ and try again."
    )

for pdf_path in pdf_files:
    print(f"\nFile: {pdf_path.name}")

    with fitz.open(pdf_path) as document:
        print(f"Pages: {len(document)}")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                print(f"\n--- Page {page_number} ---")
                print(text[:1000])
            else:
                print(f"\nPage {page_number}: no extractable text")
