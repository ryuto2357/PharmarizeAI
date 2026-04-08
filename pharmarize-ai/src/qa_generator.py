#!/usr/bin/env python3
"""
QA Dataset Generator for PharmarizeAI
Generates SQuAD-format Q&A pairs from extracted journal texts.

This script uses pattern matching and NLP heuristics to extract
question-answer pairs about Indonesian medicinal plants.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


# Question templates for different information types
QUESTION_TEMPLATES = {
    "scientific_name": [
        "Apa nama ilmiah dari {plant}?",
        "Apa nama latin {plant}?",
        "Sebutkan nama ilmiah {plant}!",
    ],
    "local_name": [
        "Apa nama lokal dari {scientific}?",
        "Tanaman {scientific} dikenal dengan nama apa?",
    ],
    "compound": [
        "Apa kandungan senyawa aktif dalam {plant}?",
        "Senyawa apa yang terkandung dalam {plant}?",
        "Apa kandungan kimia {plant}?",
        "Sebutkan senyawa yang terdapat pada {plant}!",
    ],
    "benefit": [
        "Apa manfaat {plant}?",
        "Apa khasiat {plant}?",
        "{plant} bermanfaat untuk apa?",
        "Apa kegunaan {plant} dalam pengobatan?",
    ],
    "location": [
        "Di mana {plant} tumbuh?",
        "Di wilayah mana {plant} dapat ditemukan?",
        "Dimana habitat {plant}?",
    ],
    "extraction_method": [
        "Bagaimana cara mengekstraksi {compound} dari {plant}?",
        "Metode apa yang digunakan untuk ekstraksi {plant}?",
        "Bagaimana proses ekstraksi {plant}?",
    ],
    "activity": [
        "Apa aktivitas biologis {compound}?",
        "{compound} memiliki aktivitas apa?",
        "Apa efek farmakologis {plant}?",
    ],
    "usage": [
        "Bagaimana cara penggunaan {plant} secara tradisional?",
        "Bagaimana {plant} digunakan dalam pengobatan tradisional?",
    ],
    "part_used": [
        "Bagian apa dari {plant} yang digunakan?",
        "Bagian tanaman {plant} mana yang berkhasiat?",
    ],
}

# Patterns to extract information from text
PATTERNS = {
    # Scientific name patterns: "Tanaman X (Scientific name)" or "Scientific name (X)"
    "scientific_name": [
        r'([A-Z][a-z]+\s+[a-z]+(?:\s+[A-Z][a-z\.]+)?)\s*\([^)]+\)',  # Scientific (local)
        r'([a-zA-Z]+)\s*\(([A-Z][a-z]+\s+[a-z]+(?:\s+[A-Z][a-z\.]+)?)\)',  # local (Scientific)
        r'tanaman\s+([a-z]+)\s*\(([A-Z][a-z]+\s+[a-z]+)\)',  # tanaman X (Scientific)
    ],
    
    # Compound patterns
    "compound": [
        r'mengandung\s+(?:senyawa\s+)?([a-zA-Z0-9\-,\s]+?)(?:\s+yang|\s+dengan|\s*\.|\s*,)',
        r'kandungan\s+(?:utama\s+)?(?:berupa\s+)?([a-zA-Z0-9\-,\s]+?)(?:\s+yang|\s*\.|\s*,)',
        r'senyawa\s+(?:aktif\s+)?([a-zA-Z0-9\-]+)',
        r'(?:flavonoid|alkaloid|terpenoid|saponin|tanin|fenol|kurkumin|eurycomanone|quercetin|kuersetin)',
    ],
    
    # Benefit/activity patterns
    "benefit": [
        r'(?:bermanfaat|berkhasiat|digunakan|dapat)\s+(?:untuk|sebagai)\s+([^\.]+)',
        r'(?:anti[a-z]+|antioxidant|antioksidan|antiinflamasi|antibakteri|antikanker)',
        r'(?:mengatasi|mengobati|menyembuhkan)\s+([^\.]+?)(?:\s*\.|\s*,)',
    ],
    
    # Location patterns
    "location": [
        r'(?:tumbuh|ditemukan|tersebar)\s+(?:di|pada)\s+([^\.]+?)(?:\s*\.|\s*,)',
        r'(?:wilayah|daerah|pulau)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)',
    ],
}

# Common Indonesian medicinal plants to look for
KNOWN_PLANTS = [
    ("pasak bumi", "Eurycoma longifolia"),
    ("kunyit", "Curcuma longa"),
    ("jahe", "Zingiber officinale"),
    ("temulawak", "Curcuma xanthorrhiza"),
    ("kumis kucing", "Orthosiphon aristatus"),
    ("sambiloto", "Andrographis paniculata"),
    ("mengkudu", "Morinda citrifolia"),
    ("mahkota dewa", "Phaleria macrocarpa"),
    ("daun sirsak", "Annona muricata"),
    ("pegagan", "Centella asiatica"),
    ("bajakah", "Spatholobus littoralis"),
    ("kelor", "Moringa oleifera"),
    ("lidah buaya", "Aloe vera"),
    ("binahong", "Anredera cordifolia"),
    ("sirih", "Piper betle"),
    ("kayu manis", "Cinnamomum verum"),
    ("cengkeh", "Syzygium aromaticum"),
    ("serai", "Cymbopogon citratus"),
    ("jintan hitam", "Nigella sativa"),
    ("rosella", "Hibiscus sabdariffa"),
]


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep Indonesian text
    text = re.sub(r'[^\w\s\.\,\(\)\-\:\;]', '', text)
    return text.strip()


def split_into_paragraphs(text: str, min_length: int = 200, max_length: int = 1500) -> List[str]:
    """Split text into meaningful paragraphs/contexts."""
    # Split by double newlines or section markers
    sections = re.split(r'\n\s*\n|\n(?=[A-Z0-9]+\.?\s+[A-Z])', text)
    
    paragraphs = []
    current = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Skip very short sections
        if len(section) < 100:
            current += " " + section
            continue
            
        # If section is good length, add it
        if min_length <= len(section) <= max_length:
            paragraphs.append(clean_text(section))
        elif len(section) > max_length:
            # Split long sections by sentences
            sentences = re.split(r'(?<=[\.!?])\s+', section)
            chunk = ""
            for sent in sentences:
                if len(chunk) + len(sent) < max_length:
                    chunk += " " + sent
                else:
                    if len(chunk) >= min_length:
                        paragraphs.append(clean_text(chunk))
                    chunk = sent
            if len(chunk) >= min_length:
                paragraphs.append(clean_text(chunk))
        else:
            current += " " + section
            if len(current) >= min_length:
                paragraphs.append(clean_text(current))
                current = ""
    
    if len(current) >= min_length:
        paragraphs.append(clean_text(current))
    
    return paragraphs


def find_answer_in_context(context: str, answer: str) -> int:
    """Find the starting position of answer in context."""
    # Try exact match first
    pos = context.find(answer)
    if pos != -1:
        return pos
    
    # Try case-insensitive
    pos = context.lower().find(answer.lower())
    if pos != -1:
        # Return the actual position in original text
        return pos
    
    return -1


def extract_plant_info(context: str) -> Dict:
    """Extract plant-related information from context."""
    info = {
        "plants": [],
        "compounds": [],
        "benefits": [],
        "locations": [],
    }
    
    context_lower = context.lower()
    
    # Find known plants
    for local, scientific in KNOWN_PLANTS:
        if local in context_lower or scientific.lower() in context_lower:
            info["plants"].append((local, scientific))
    
    # Extract compounds - more specific patterns
    compound_patterns = [
        r'(eurycomanone|eurycomalactone|eurycomaoside|niloticin)',
        r'(kurkumin|curcumin|kurkuminoid)',
        r'(quercetin|kuersetin|quercitrin)',
        r'(flavonoid|alkaloid|saponin|tanin|terpenoid|fenol|polifenol|steroid|triterpenoid)',
        r'(andrographolide|gingerol|shogaol|xanthorrhizol)',
        r'(antosianin|karotenoid|vitamin\s+[A-Z])',
    ]
    for pattern in compound_patterns:
        matches = re.findall(pattern, context_lower)
        info["compounds"].extend(matches)
    
    # Extract benefits
    benefit_patterns = [
        r'(?:sebagai|untuk)\s+(anti[a-z]+)',
        r'(antioksidan|antiinflamasi|antibakteri|antikanker|antidiabetes|antihipertensi)',
        r'(?:mengatasi|mengobati)\s+([a-z\s]+?)(?:\.|,)',
    ]
    for pattern in benefit_patterns:
        matches = re.findall(pattern, context_lower)
        info["benefits"].extend(matches)
    
    # Clean duplicates
    info["compounds"] = list(set(info["compounds"]))
    info["benefits"] = list(set(info["benefits"]))
    
    return info


def generate_qa_pairs(context: str, doc_id: str, start_qid: int) -> List[Dict]:
    """Generate Q&A pairs from a context paragraph."""
    qa_pairs = []
    qid = start_qid
    
    info = extract_plant_info(context)
    
    # Generate questions about plants found
    for local_name, scientific_name in info["plants"]:
        # Question about scientific name
        if scientific_name.lower() in context.lower():
            pos = find_answer_in_context(context, scientific_name)
            if pos == -1:
                # Try to find in original case
                for match in re.finditer(re.escape(scientific_name), context, re.IGNORECASE):
                    pos = match.start()
                    scientific_name = match.group()
                    break
            
            if pos != -1:
                question = random.choice(QUESTION_TEMPLATES["scientific_name"]).format(plant=local_name)
                qa_pairs.append({
                    "question": question,
                    "id": f"{doc_id}_q{qid:03d}",
                    "answers": [{"text": scientific_name, "answer_start": pos}],
                    "is_impossible": False
                })
                qid += 1
    
    # Generate questions about compounds
    for compound in info["compounds"][:3]:  # Limit to 3 compounds per context
        if len(compound) < 5:  # Skip very short matches (was 4, now 5)
            continue
        
        # Skip generic words that aren't actual compound names
        skip_words = {'yang', 'dan', 'atau', 'dari', 'untuk', 'dengan', 'pada', 'bahan', 'hasil'}
        if compound.lower() in skip_words:
            continue
            
        # Find the compound in context
        match = re.search(re.escape(compound), context, re.IGNORECASE)
        if match:
            pos = match.start()
            actual_text = match.group()
            
            # Find which plant this is about
            plant_name = "tanaman ini"
            for local, _ in info["plants"]:
                if local in context.lower():
                    plant_name = local
                    break
            
            question = random.choice(QUESTION_TEMPLATES["compound"]).format(plant=plant_name)
            qa_pairs.append({
                "question": question,
                "id": f"{doc_id}_q{qid:03d}",
                "answers": [{"text": actual_text, "answer_start": pos}],
                "is_impossible": False
            })
            qid += 1
    
    # Generate questions about benefits/activities
    for benefit in info["benefits"][:2]:  # Limit to 2 benefits per context
        if len(benefit) < 5:
            continue
            
        match = re.search(re.escape(benefit), context, re.IGNORECASE)
        if match:
            pos = match.start()
            actual_text = match.group()
            
            plant_name = "tanaman ini"
            for local, _ in info["plants"]:
                if local in context.lower():
                    plant_name = local
                    break
            
            question = random.choice(QUESTION_TEMPLATES["benefit"]).format(plant=plant_name)
            qa_pairs.append({
                "question": question,
                "id": f"{doc_id}_q{qid:03d}",
                "answers": [{"text": actual_text, "answer_start": pos}],
                "is_impossible": False
            })
            qid += 1
    
    # Extract longer answer spans using sentence patterns - more specific
    sentence_patterns = [
        # Pattern: "X mengandung senyawa Y"
        (r'([A-Z][a-z]+[^\.]{10,80}mengandung\s+(?:senyawa\s+)?[a-zA-Z\-]+[^\.]{5,50}\.)', "compound"),
        # Pattern: "X bermanfaat/berkhasiat untuk Y" 
        (r'([A-Z][a-z]+[^\.]{10,80}(?:bermanfaat|berkhasiat)\s+(?:untuk|sebagai)\s+[^\.]{10,60}\.)', "benefit"),
        # Pattern: "X tumbuh di wilayah Y"
        (r'([A-Z][a-z]+[^\.]{10,80}(?:tumbuh|ditemukan|tersebar)\s+(?:di|pada)\s+[^\.]{10,60}\.)', "location"),
    ]
    
    for pattern, qtype in sentence_patterns:
        matches = list(re.finditer(pattern, context))
        for match in matches[:1]:  # One per pattern type
            answer_text = match.group(1).strip()
            if len(answer_text) > 200:  # Skip too long answers
                continue
            
            pos = match.start(1)
            
            # Determine plant name for question
            plant_name = "tanaman ini"
            for local, _ in info["plants"]:
                if local in answer_text.lower():
                    plant_name = local
                    break
            
            if qtype in QUESTION_TEMPLATES:
                question = random.choice(QUESTION_TEMPLATES[qtype]).format(
                    plant=plant_name, 
                    compound=info["compounds"][0] if info["compounds"] else "senyawa"
                )
                qa_pairs.append({
                    "question": question,
                    "id": f"{doc_id}_q{qid:03d}",
                    "answers": [{"text": answer_text, "answer_start": pos}],
                    "is_impossible": False
                })
                qid += 1
    
    return qa_pairs


def process_document(file_path: str, doc_id: str) -> Dict:
    """Process a single document and generate QA data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    title = Path(file_path).stem
    paragraphs_data = []
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(text)
    
    total_qas = 0
    for i, para in enumerate(paragraphs):
        if len(para) < 200:  # Skip short paragraphs
            continue
        
        # Generate QA pairs
        qas = generate_qa_pairs(para, f"{doc_id}_p{i}", total_qas)
        
        if qas:  # Only add paragraphs with QA pairs
            paragraphs_data.append({
                "context": para,
                "qas": qas
            })
            total_qas += len(qas)
    
    return {
        "title": title,
        "paragraphs": paragraphs_data,
        "total_qas": total_qas
    }


def generate_dataset(input_dir: str, output_path: str) -> Dict:
    """Generate full QA dataset from all documents."""
    input_path = Path(input_dir)
    txt_files = sorted(input_path.glob("*.txt"))
    
    all_data = []
    stats = {
        "total_documents": len(txt_files),
        "documents_with_qa": 0,
        "total_paragraphs": 0,
        "total_qa_pairs": 0,
    }
    
    print(f"Processing {len(txt_files)} documents...")
    
    for i, txt_file in enumerate(txt_files, 1):
        doc_id = f"doc_{i:03d}"
        print(f"[{i}/{len(txt_files)}] {txt_file.name}...", end=" ")
        
        try:
            doc_data = process_document(str(txt_file), doc_id)
            
            if doc_data["paragraphs"]:
                all_data.append(doc_data)
                stats["documents_with_qa"] += 1
                stats["total_paragraphs"] += len(doc_data["paragraphs"])
                stats["total_qa_pairs"] += doc_data["total_qas"]
                print(f"✓ {doc_data['total_qas']} QAs")
            else:
                print("(no QA extracted)")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Create final dataset
    dataset = {
        "data": all_data,
        "stats": stats
    }
    
    # Save dataset
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Dataset generated: {output_file}")
    print(f"  Documents processed: {stats['total_documents']}")
    print(f"  Documents with QA:   {stats['documents_with_qa']}")
    print(f"  Total paragraphs:    {stats['total_paragraphs']}")
    print(f"  Total QA pairs:      {stats['total_qa_pairs']}")
    print(f"{'='*50}")
    
    return stats


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / "data" / "processed"
    output_path = script_dir / "data" / "qa_dataset.json"
    
    generate_dataset(str(input_dir), str(output_path))
