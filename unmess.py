import os
import subprocess
import json
import PyPDF2
import fitz
import nltk
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision import models
import requests

nltk.download('punkt')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(BASE_DIR, "IN")
OUT_DIR = os.path.join(BASE_DIR, "OUT")

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained image recognition model and move it to the GPU if available
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

# ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

def process_file(file_name):
    file_path = os.path.join(IN_DIR, file_name)
    if os.path.isfile(file_path):
        print(f"Analyzuji soubor: {file_name}")
        
        # Initialize metadata structure
        data = {
            "file_path": file_path,  # Add full file path to metadata
            "text_metadata": None,
            "metadata": None,
            "headings": [],
            "intro_text": "",
            "image_analysis": "",
            "image_metadata": None
        }
        
        # Attempt to read as a text file
        text_metadata = extract_text_metadata(file_path)
        if text_metadata:
            data["text_metadata"] = text_metadata
        else:
            # Process as PDF or image
            if file_name.lower().endswith('.pdf'):
                data["metadata"] = extract_metadata(file_path)
                data["headings"] = extract_headings(file_path)
                data["intro_text"] = extract_intro_text(file_path)
                data["image_analysis"] = extract_and_analyze_images(file_path)
            elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                data["image_metadata"] = extract_image_metadata(file_path)

        # Debug: Print data to ensure it's not empty
        print("Data to be written to JSON:", data)

        # Uložení do JSON souboru
        output_file_path = os.path.join(OUT_DIR, f"{os.path.splitext(file_name)[0]}.json")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=4)

        print(f"Soubor: {file_name} byl zpracován a uložen jako JSON.")
        print("-" * 40)

def extract_text_metadata(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        headers = []
        comments = []
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('#'):  # Assuming comments start with '#'
                comments.append(stripped_line)
            elif ':' in stripped_line:  # Assuming headers have a ':' character
                headers.append(stripped_line)

        # Limit to 10 comments: first 3, middle, and last
        total_comments = len(comments)
        if total_comments > 10:
            comments = comments[:3] + comments[total_comments//2-3:total_comments//2+2] + comments[-3:]

        return {
            "headers": headers[:10],  # Limit to first 10 headers
            "comments": comments
        }
    except Exception as e:
        print(f"Chyba při čtení textového souboru {file_path}: {e}")
        return None

def extract_image_metadata(file_path):
    try:
        # Call exiftool to extract metadata
        result = subprocess.run(['exiftool', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            metadata = {}
            for line in result.stdout.splitlines():
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key in ["File Name", "Create Date"]:
                    metadata[key] = value
            return metadata
        else:
            print(f"Chyba při čtení metadat ze souboru {file_path}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Chyba při zpracování metadat ze souboru {file_path}: {e}")
        return None

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
    except Exception as e:
        print(f"Chyba při čtení souboru {file_path}: {e}")
        return ""

def extract_headings(file_path):
    try:
        doc = fitz.open(file_path)
        headings = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] > 12:
                                heading_text = span["text"].strip()
                                if heading_text and heading_text not in headings:
                                    headings.append(heading_text)
                                    if len(headings) >= 10:  # Limit to 10 headings
                                        return headings
        return headings
    except Exception as e:
        print(f"Chyba při extrakci nadpisů ze souboru {file_path}: {e}")
        return []

import re

def extract_intro_text(file_path):
    text = extract_text_from_pdf(file_path)
    if text:
        # Split the text into words
        words = text.split()
        total_words = len(words)
        
        # Determine indices for middle words
        middle_start = max(0, total_words // 2 - 15)
        middle_end = min(total_words, middle_start + 30)
        
        # Select words
        intro_words = (
            words[:30] +  # First 30 words
            words[middle_start:middle_end] +  # 30 words from the middle
            words[-30:]  # Last 30 words
        )
        
        return ' '.join(intro_words)
    else:
        return ""

def extract_metadata(file_path):
    try:
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            metadata = pdf.metadata
            return metadata
    except Exception as e:
        print(f"Chyba při extrakci metadat ze souboru {file_path}: {e}")
        return None

def extract_and_analyze_images(file_path):
    try:
        doc = fitz.open(file_path)
        image_info = []
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Process image
                ocr_text, description = process_image(image_bytes)
                image_info.append(f"Page {page_number + 1} Image {img_index + 1}: OCR Text: {ocr_text}, Description: {description}")
        return "\n".join(image_info)
    except Exception as e:
        print(f"Chyba při zpracování obrázků ze souboru {file_path}: {e}")
        return ""

def process_image(image_bytes):
    # Convert image bytes to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Perform OCR
    ocr_text = pytesseract.image_to_string(image)
    
    # Describe image content
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).to(device)  # Move tensor to GPU
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    description = labels[predicted_idx.item()]

    return ocr_text, description

# Define the number of threads you want to use
num_threads = 3

# Use ThreadPoolExecutor with the specified number of threads
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(process_file, os.listdir(IN_DIR))

print("Analýza dokončena.")
