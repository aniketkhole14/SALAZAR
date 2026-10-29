from app.ingestion.chunker import chunk_text

sample = " ".join(f"word{i}" for i in range(500))
chunks = chunk_text(sample, page_number=2)

assert len(chunks) > 1
assert all(chunk.page_number == 2 for chunk in chunks)
assert all(chunk.text for chunk in chunks)

print("Chunks created:", len(chunks))
print("First chunk words:", len(chunks[0].text.split()))
print("Second chunk words:", len(chunks[1].text.split()))
print("First chunk page:", chunks[0].page_number)
print("Chunker test passed")
