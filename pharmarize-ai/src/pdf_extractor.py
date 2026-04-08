#!/usr/bin/env python3
"""
PDF to Text Extractor for PharmarizeAI
Converts PDF journal files to structured text data for training.
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text content from a PDF file."""
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            text_parts.append(text.strip())
    
    doc.close()
    return "\n\n".join(text_parts)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    # Join lines, preserving paragraph breaks (double newlines)
    result = []
    prev_empty = False
    
    for line in cleaned_lines:
        if not line:
            prev_empty = True
        else:
            if prev_empty and result:
                result.append('')
            result.append(line)
            prev_empty = False
    
    return '\n'.join(result)


def process_all_pdfs(input_dir: str, output_dir: str) -> dict:
    """Process all PDFs in input directory and save to output directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = sorted(input_path.glob("*.pdf"))
    
    all_documents = []
    stats = {"total": len(pdf_files), "success": 0, "failed": 0, "errors": []}
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            raw_text = extract_text_from_pdf(str(pdf_file))
            cleaned_text = clean_text(raw_text)
            
            if not cleaned_text.strip():
                print(f"  Warning: No text extracted from {pdf_file.name}")
                stats["failed"] += 1
                stats["errors"].append({"file": pdf_file.name, "error": "No text extracted"})
                continue
            
            # Create document entry
            doc_entry = {
                "id": f"doc_{i:03d}",
                "source_file": pdf_file.name,
                "title": pdf_file.stem,
                "content": cleaned_text,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split())
            }
            
            all_documents.append(doc_entry)
            
            # Save individual text file
            txt_output = output_path / f"{pdf_file.stem}.txt"
            with open(txt_output, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            stats["success"] += 1
            print(f"  Extracted {doc_entry['word_count']} words")
            
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")
            stats["failed"] += 1
            stats["errors"].append({"file": pdf_file.name, "error": str(e)})
    
    # Save combined JSON with all documents
    combined_output = output_path / "all_documents.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump({
            "documents": all_documents,
            "stats": stats
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Success: {stats['success']}/{stats['total']}")
    print(f"  Failed:  {stats['failed']}/{stats['total']}")
    print(f"  Output:  {output_path}")
    print(f"{'='*50}")
    
    return stats


if __name__ == "__main__":
    # Default paths relative to project root
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / "data" / "raw_journals"
    output_dir = script_dir / "data" / "processed"
    
    process_all_pdfs(str(input_dir), str(output_dir))
