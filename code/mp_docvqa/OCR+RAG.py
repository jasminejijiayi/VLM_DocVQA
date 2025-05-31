from openai import OpenAI
from datasets import load_from_disk
import json
import argparse
import base64
from tqdm import tqdm
import os
import wandb
from datetime import datetime
import time
import numpy as np
from PIL import Image
import io

# Import pytesseract for OCR text extraction
import pytesseract
import cv2

# Import Whoosh for BM25 retrieval
import whoosh
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import scoring
import shutil

# Set up OpenAI client
client = OpenAI(
    base_url="http://47.242.151.133:24576/v1/",
    api_key="ml2025",
)

# Configure pytesseract (path to tesseract executable if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment if needed


def load_data(path):
    '''
    Load data from disk
    Args:
        path: str, the path to the data
    Returns:
        ds: Dataset, the data
    '''
    ds = load_from_disk(path)
    return ds


def extract_text_with_pytesseract(img):
    '''
    Extract text from an image using pytesseract
    Args:
        img: PIL Image, the image to extract text from
    Returns:
        text: str, the extracted text
    '''
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
            
        # Preprocess image for better OCR results
        # Convert to grayscale if not already
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        # Apply threshold to get black and white image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run OCR with pytesseract
        # Use PSM 6 (Assume a single uniform block of text) for better results
        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=config)
        
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


def setup_bm25_index(texts, index_dir="./bm25_index"):
    '''
    Set up BM25 index for text retrieval
    Args:
        texts: list of dicts with 'id' and 'text' keys
        index_dir: directory to store the index
    Returns:
        index_dir: path to the created index
    '''
    # Create a fresh index directory
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir)
    
    # Define schema
    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    
    # Create index
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    
    # Add documents to index
    for doc in texts:
        writer.add_document(id=str(doc['id']), content=doc['text'])
    
    # Commit changes
    writer.commit()
    
    return index_dir


def retrieve_with_bm25(query, index_dir="./bm25_index", top_k=3):
    '''
    Retrieve relevant documents using BM25
    Args:
        query: str, the query text
        index_dir: directory containing the index
        top_k: number of documents to retrieve
    Returns:
        results: list of retrieved documents
    '''
    # Open index
    ix = open_dir(index_dir)
    
    # Create searcher with BM25F scoring
    searcher = ix.searcher(weighting=scoring.BM25F())
    
    # Parse query
    parser = QueryParser("content", ix.schema)
    q = parser.parse(query)
    
    # Search
    results = searcher.search(q, limit=top_k)
    
    # Extract results
    retrieved_docs = []
    for hit in results:
        retrieved_docs.append({
            'id': hit['id'],
            'content': hit['content'],
            'score': hit.score
        })
    
    # Close searcher
    searcher.close()
    
    return retrieved_docs


def preprocess_image(example):
    '''
    Preprocess images, extract text with OCR, and retrieve relevant content
    Args:
        example: dict, the example
    Returns:
        dict: containing processed image and RAG context
    '''
    # Collect all images using a loop
    images = []
    for i in range(1, 21):
        img_key = f"image_{i}"
        if img_key in example and example[img_key] is not None:
            images.append((i, example[img_key]))
    
    # Print the number of images found
    print(f"Found {len(images)} images in the example")
    
    # If no images found, return None
    if not images:
        print("No images found in the example")
        return None
    
    # Get the query
    query = example['question']
    print(f"Query: {query}")
    
    # [1] Process multiple PNG images
    processed_images = []
    
    # [2] Extract text using PaddleOCR
    image_texts = []
    
    for idx, img in images:
        # Convert image to PIL Image if it's not already
        if not isinstance(img, Image.Image):
            try:
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img))
                else:
                    img = Image.open(img)
            except Exception as e:
                print(f"Error opening image {idx}: {e}")
                continue
        
        # Convert to grayscale for better OCR
        img_gray = img.convert('L')
        processed_images.append((idx, img_gray))
        
        # Extract text using pytesseract
        text = extract_text_with_pytesseract(img_gray)
        print(f"Image {idx} OCR text length: {len(text)}")
        
        # [3] Build text blocks (one per image)
        if text.strip():
            image_texts.append({
                'id': str(idx),
                'text': text,
                'image': img_gray
            })
    
    # If we have extracted texts, proceed with BM25 retrieval
    if image_texts:
        # Create documents for BM25 indexing
        documents = [{'id': doc['id'], 'text': doc['text']} for doc in image_texts]
        
        # [4] Set up BM25 index and retrieve relevant documents
        index_dir = setup_bm25_index(documents)
        retrieved_docs = retrieve_with_bm25(query, index_dir)
        
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        if retrieved_docs:
            # Get the IDs of the top retrieved images
            top_image_ids = [int(doc['id']) for doc in retrieved_docs]
            
            # Get the corresponding images and texts
            top_images = []
            rag_texts = []
            
            for doc in retrieved_docs:
                doc_id = int(doc['id'])
                # Find the corresponding image
                for idx, img in processed_images:
                    if idx == doc_id:
                        top_images.append(img)
                        break
                # Add the text to RAG context
                rag_texts.append(doc['content'])
            
            # [5] Concatenate context for RAG
            rag_context = "\n\n".join(rag_texts)
            
            # Concatenate the top images
            if len(top_images) == 1:
                final_image = top_images[0]
            elif len(top_images) > 1:
                # Determine the concatenation direction based on image dimensions
                # Use the longer edge for concatenation to create more balanced outputs
                widths, heights = zip(*(img.size for img in top_images))
                
                if max(widths) > max(heights):
                    # Concatenate vertically if images are wider
                    total_width = max(widths)
                    total_height = sum(heights)
                    concat_img = Image.new('L', (total_width, total_height))
                    
                    y_offset = 0
                    for img in top_images:
                        # Center the image horizontally
                        x_offset = (total_width - img.width) // 2
                        concat_img.paste(img, (x_offset, y_offset))
                        y_offset += img.height
                else:
                    # Concatenate horizontally if images are taller
                    total_width = sum(widths)
                    total_height = max(heights)
                    concat_img = Image.new('L', (total_width, total_height))
                    
                    x_offset = 0
                    for img in top_images:
                        # Center the image vertically
                        y_offset = (total_height - img.height) // 2
                        concat_img.paste(img, (x_offset, y_offset))
                        x_offset += img.width
                
                final_image = concat_img
            else:
                # Fallback to first image if something went wrong
                final_image = processed_images[0][1] if processed_images else None
                
            if final_image is None:
                return None
                
            # Convert back to RGB for compatibility
            final_image = final_image.convert('RGB')
            
            print(f"Selected {len(top_images)} top images based on BM25 retrieval")
            print(f"RAG context length: {len(rag_context)} characters")
            
            return {
                'image': final_image,
                'rag_context': rag_context
            }
    
    # Fallback: If no text extracted or retrieval failed, use the first image
    print("Using fallback to first image")
    first_image = processed_images[0][1] if processed_images else None
    
    if first_image is None:
        return None
        
    # Convert to RGB for compatibility
    first_image = first_image.convert('RGB')
    
    return {
        'image': first_image,
        'rag_context': ""
    }


def generate_answer(example):
    '''
    Generate answer for the an example using OCR and RAG
    Args:
        example: dict, the example
    Returns:
        answer: str, the answer
    '''
    # [1-4] Preprocess images, extract text with OCR, and retrieve relevant content
    result = preprocess_image(example)

    if result is None:
        print("No image data available")
        return None
    
    # Extract image and RAG context
    image = result['image']
    rag_context = result['rag_context']
    
    # Print RAG context length for debugging
    if rag_context:
        print(f"Using RAG context with {len(rag_context)} characters")
    else:
        print("No RAG context available, using image only")

    # Convert image to base64
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, "tmp_image.png")
    image.save(tmp_file, format="PNG")

    with open(tmp_file, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_image = f"data:image;base64,{encoded_image_text}"

    # Get the question
    question = example['question']
    
    # [5] Create prompt with RAG context if available
    if rag_context:
        # Combine RAG context with question
        text = f"""Context from the document:
{rag_context}

Question: {question}

Based on the context provided and the image, please answer the question. Only return the answer, no other words."""
    else:
        # Use only the question if no RAG context
        text = f"{question}\nOnly return the answer, no other words."

    # Log the prompt for debugging
    print(f"Sending prompt to Qwen model with {len(text)} characters")
    
    # [6] Send request to Qwen model and return answer
    try:
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about documents accurately based on the provided context and image."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                },
            ],
        )
        answer = chat_response.choices[0].message.content
        print(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None


def evaluate_results(results):
    '''
    Evaluate the results
    Args:
        results: list, the results
    Returns:
        score: float, the score
    '''
    # Calculate the score
    score = 0
    for result in results:
        if result["generation"].lower() in result["answers"].lower():
            score += 1
    return round(score / len(results), 2)


def main(args):
    # Initialize wandb if tracking is enabled
    if args.use_wandb:
        run_name = f"baseline_eval_{datetime.now().strftime('%Y%m%d_%H%M')}" if args.eval_only else f"baseline_{datetime.now().strftime('%Y%m%d_%H%M')}"
        run = wandb.init(
            project="mp-docvqa-optimization",
            name=run_name,
            config={
                "data_path": args.data_path,
                "model": "Qwen2.5-VL-3B-Instruct",
                "experiment_type": "baseline",
                "image_selection": "first_image",
                "eval_only": args.eval_only
            }
        )
    
    if not args.eval_only:
        # Load data
        ds = load_data(args.data_path)
        
        if args.use_wandb:
            wandb.log({"dataset_size": len(ds)})

        # Generate
        results = []
        correct_count = 0
        total_time = 0
        
        for i, example in enumerate(tqdm(ds, desc="Generating answers", total=len(ds), leave=True, position=0)):
            question = example['question']
            print(f"Example {i+1}/{len(ds)}: {question}")
            
            # Time the generation
            start_time = time.time()
            answer = generate_answer(example)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            print(f"Answer: {answer} (took {elapsed:.2f}s)")
            
            # Check if answer is correct
            is_correct = answer.lower() in example['answers'].lower()
            if is_correct:
                correct_count += 1
                print("✓ Correct")
            else:
                print("✗ Incorrect")
                
            # Log additional MP-DocVQA specific info
            answer_page_idx = example.get('answer_page_idx', None)
            used_page_idx = "1"  # Since baseline uses the first image
            used_correct_page = (answer_page_idx == used_page_idx) if answer_page_idx is not None else False
            
            results.append({
                "generation": answer, 
                "answers": example['answers'],
                "answer_page_idx": answer_page_idx,
                "used_page_idx": used_page_idx,
                "used_correct_page": used_correct_page
            })
            
            # Log to wandb if enabled
            if args.use_wandb:
                # Log individual example metrics
                wandb.log({
                    "question": question,
                    "prediction": answer,
                    "ground_truth": example['answers'],
                    "correct": is_correct,
                    "example_idx": i,
                    "response_time": elapsed,
                    "answer_page_idx": answer_page_idx,
                    "used_page_idx": used_page_idx,
                    "used_correct_page": used_correct_page
                })
                
                # Log running accuracy
                current_accuracy = correct_count / (i + 1)
                wandb.log({
                    "running_accuracy": current_accuracy,
                    "examples_processed": i + 1
                })
                
                # Log image for the first 10 examples to save space
                if i < 10:
                    try:
                        # Get the preprocessed image
                        image = preprocess_image(example)
                        # Log the image with question and answer as caption
                        wandb.log({f"image_{i}": wandb.Image(
                            image, 
                            caption=f"Q: {question}, A: {answer}, Correct: {is_correct}"
                        )})
                    except Exception as e:
                        print(f"Error logging image: {e}")

        # Evaluate
        pass_rate = evaluate_results(results)
        avg_time = total_time / len(ds) if len(ds) > 0 else 0
        print(f"Pass rate: {pass_rate}")
        print(f"Average response time: {avg_time:.2f}s")

        # Calculate additional MP-DocVQA specific metrics
        correct_page_count = sum(1 for r in results if r.get('used_correct_page', False))
        correct_page_rate = round(correct_page_count / len(results), 2) if len(results) > 0 else 0
        print(f"Used correct page rate: {correct_page_rate}")
        
        # Calculate accuracy when using correct page vs. incorrect page
        correct_page_results = [r for r in results if r.get('used_correct_page', False)]
        incorrect_page_results = [r for r in results if not r.get('used_correct_page', False)]
        
        correct_page_accuracy = evaluate_results(correct_page_results) if correct_page_results else 0
        incorrect_page_accuracy = evaluate_results(incorrect_page_results) if incorrect_page_results else 0
        
        print(f"Accuracy when using correct page: {correct_page_accuracy}")
        print(f"Accuracy when using incorrect page: {incorrect_page_accuracy}")

        # Log final metrics to wandb
        if args.use_wandb:
            wandb.log({
                "final_pass_rate": pass_rate,
                "average_response_time": avg_time,
                "total_examples": len(ds),
                "correct_examples": correct_count,
                "used_correct_page_rate": correct_page_rate,
                "correct_page_accuracy": correct_page_accuracy,
                "incorrect_page_accuracy": incorrect_page_accuracy
            })

        # Save results to disk
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(args.output_path, "w") as f:
            json.dump(results, f)
    else:
        # Load results
        with open(args.output_path, "r") as f:
            results = json.load(f)
        # Evaluate
        pass_rate = evaluate_results(results)
        print(f"Pass rate: {pass_rate}")
        
        # Log evaluation results to wandb if enabled
        if args.use_wandb:
            wandb.log({"eval_pass_rate": pass_rate})
    
    # Finish wandb run if it was started
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="code/data/mp_docvqa_100")
    parser.add_argument("--output_path", type=str, default="code/mp_docvqa/results/OCR+RAG.json")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb_project", type=str, default="mp-docvqa-optimization", help="Weights & Biases project name")
    args = parser.parse_args()

    main(args)
