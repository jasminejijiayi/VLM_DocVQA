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
from PIL import Image
import io
import numpy as np

client = OpenAI(
    base_url="http://47.242.151.133:24576/v1/",
    api_key="ml2025",
)


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


def preprocess_image(example):
    '''
    Preprocess the image for better performance by converting to grayscale
    Args:
        example: dict, the example
    Returns:
        image: Image, the grayscale image optimized for Qwen multimodal model
    '''
    # Get the image from the example
    image = None
    
    try:
        # Handle different image formats
        if isinstance(example["image"], Image.Image):
            # Already a PIL Image
            image = example["image"]
        elif isinstance(example["image"], bytes):
            # Bytes data
            image = Image.open(io.BytesIO(example["image"]))
        elif isinstance(example["image"], str) and os.path.exists(example["image"]):
            # File path
            image = Image.open(example["image"])
        else:
            # Default case
            image = example["image"]
            
        # Convert to grayscale (L mode = 8-bit pixels, black and white)
        if image.mode != 'L':
            image = image.convert('L')
            
        # Enhance contrast to improve readability of text in documents
        # This is particularly helpful for document images
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Optional: Apply slight sharpening to make text more readable
        from PIL import ImageFilter
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if the image is too large (optional, based on model requirements)
        max_dim = 1024  # Maximum dimension for Qwen model
        if image.width > max_dim or image.height > max_dim:
            # Calculate new dimensions while preserving aspect ratio
            if image.width > image.height:
                new_width = max_dim
                new_height = int(image.height * (max_dim / image.width))
            else:
                new_height = max_dim
                new_width = int(image.width * (max_dim / image.height))
            
            # Resize using high-quality resampling
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Print debug info
        print(f"Converted image to grayscale: size={image.size}, mode={image.mode}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        # Create a blank grayscale image as fallback
        image = Image.new('L', (800, 600), color=255)
    
    return image


def generate_answer(example):
    '''
    Generate answer for the an example  
    Args:
        example: dict, the example
    Returns:
        answer: str, the answer
    '''
    # Preprocess the image
    image = preprocess_image(example)

    # Convert image to base64 with optimized settings for grayscale images
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, "tmp_image.png")
    
    # Save as optimized PNG with compression level 9 (maximum)
    image.save(tmp_file, format="PNG", optimize=True, compress_level=9)

    with open(tmp_file, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_image = f"data:image;base64,{encoded_image_text}"

    # Prompt
    text = f"{example['question']}\nOnly return the answer, no other words.\n Attention: Use a space character as the separator between values for any single type answers!"

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant.Attention: Use a space character as the separator between values for any single type answers!"},
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
    return chat_response.choices[0].message.content


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
        lower_answers = [answer.lower().strip() for answer in result["answers"]]
        if result["generation"].lower().strip() in lower_answers:
            score += 1
    return round(score / len(results), 2)


def main(args):
    # Initialize wandb if tracking is enabled
    if args.use_wandb:
        run_name = f"baseline_eval_{datetime.now().strftime('%Y%m%d_%H%M')}" if args.eval_only else f"baseline_{datetime.now().strftime('%Y%m%d_%H%M')}"
        run = wandb.init(
            project="docvqa-optimization",
            name=run_name,
            config={
                "data_path": args.data_path,
                "model": "Qwen2.5-VL-3B-Instruct",
                "experiment_type": "baseline",
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
            is_correct = answer.lower().strip() in [a.lower().strip() for a in example['answers']]
            if is_correct:
                correct_count += 1
                print("✓ Correct")
            else:
                print("✗ Incorrect")
            
            results.append({"generation": answer, "answers": example['answers']})
            
            # Log to wandb if enabled
            if args.use_wandb:
                # Log individual example metrics
                wandb.log({
                    "question": question,
                    "prediction": answer,
                    "ground_truth": example['answers'],
                    "correct": is_correct,
                    "example_idx": i,
                    "response_time": elapsed
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

        # Log final metrics to wandb
        if args.use_wandb:
            wandb.log({
                "final_pass_rate": pass_rate,
                "average_response_time": avg_time,
                "total_examples": len(ds),
                "correct_examples": correct_count
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
    parser.add_argument("--data_path", type=str, default="code/data/docvqa_100")
    parser.add_argument("--output_path", type=str, default="code/docvqa/results/methodx.json")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb_project", type=str, default="docvqa-optimization", help="Weights & Biases project name")
    args = parser.parse_args()

    main(args)
