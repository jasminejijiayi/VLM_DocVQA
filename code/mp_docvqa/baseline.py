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
    Preprocess the image for better performance
    Args:
        example: dict, the example
    Returns:
        image: Image, the image
    '''
    # Choose the first image
    image = example["image_1"]

    # TODO: Choose the true image

    # TODO: More methods

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

    # Convert image to base64
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, "tmp_image.png")
    image.save(tmp_file, format="PNG")

    with open(tmp_file, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_image = f"data:image;base64,{encoded_image_text}"

    # Prompt
    text = f"{example['question']}\nOnly return the answer, no other words."

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
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
            used_page_idx = 1  # Since baseline uses the first image
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
    parser.add_argument("--output_path", type=str, default="code/mp_docvqa/results/baseline.json")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb_project", type=str, default="mp-docvqa-optimization", help="Weights & Biases project name")
    args = parser.parse_args()

    main(args)
