"""
Evaluation script for VLM
Evaluates model on test set and computes full VQA metrics
Supports: Local metrics (9 types) + GPT-4o evaluation
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from src.models import VLMModel, create_vlm_model
from src.data import VinternProcessor

# Import metrics from metrics/ module
FULL_METRICS_AVAILABLE = False
ScoreCalculator = None
compute_all_data = None

try:
    # Change to metrics directory for proper imports
    import importlib.util
    
    metrics_dir = PROJECT_ROOT / "metrics"
    
    # Add metrics dir and its subdirs to path
    sys.path.insert(0, str(metrics_dir))
    for subdir in metrics_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            sys.path.insert(0, str(subdir))
    
    # Now import
    from compute_score import ScoreCalculator, compute_all_data
    FULL_METRICS_AVAILABLE = True
    print("✓ Full metrics module loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import full metrics module: {e}")
    print("Using basic metrics only.")


def load_config(config_path: str) -> Dict:
    """Load YAML config"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(
    model_path: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Load model and tokenizer from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        config_path: Optional config file path
        device: Device to load model on
        
    Returns:
        model, tokenizer, processor
    """
    # Try to load config from model_path or use provided config
    config = {}
    config_file = Path(model_path) / "config.yaml"
    if config_file.exists():
        config = load_config(str(config_file))
    elif config_path:
        config = load_config(config_path)
    
    model_config = config.get("model", {})
    
    # Determine LLM type
    llm_type = model_config.get("llm_type", "qwen2-0.5b")
    
    # Get model name from registry
    from src.models.registry import ModelRegistry
    llm_config = ModelRegistry.get_llm_config(llm_type)
    llm_model_name = model_config.get("llm_model_name") or llm_config["model_name"]
    
    # Load tokenizer
    print(f"Loading tokenizer: {llm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        trust_remote_code=True,
    )
    
    # Load model
    print(f"Loading model from: {model_path}")
    
    # Check if it's a full model or just LoRA adapter
    lora_adapter_path = Path(model_path) / "lora_adapter"
    
    if lora_adapter_path.exists():
        # Load base model + LoRA adapter
        model = create_vlm_model({
            "vision_encoder_type": model_config.get("vision_encoder_type", "internvit"),
            "vision_model_name": model_config.get("vision_model_name"),
            "vision_hidden_size": model_config.get("vision_hidden_size", 1024),
            "image_size": model_config.get("image_size", 448),
            "projector_type": model_config.get("projector_type", "mlp"),
            "llm_type": llm_type,
            "llm_model_name": llm_model_name,
            "freeze_vision": True,
            "freeze_llm": True,
            "torch_dtype": model_config.get("torch_dtype", "float16"),
        })
        
        # Load LoRA adapter
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, str(lora_adapter_path))
        
        # Load non-LoRA weights if exist
        non_lora_path = Path(model_path) / "model" / "non_lora_weights.pt"
        if non_lora_path.exists():
            weights = torch.load(non_lora_path, map_location="cpu")
            if weights.get("vision_encoder_state"):
                model.vision_encoder.load_state_dict(weights["vision_encoder_state"])
            if weights.get("projector_state"):
                model.projector.load_state_dict(weights["projector_state"])
    else:
        # Try loading as full model
        model = create_vlm_model({
            "vision_encoder_type": model_config.get("vision_encoder_type", "internvit"),
            "vision_model_name": model_config.get("vision_model_name"),
            "vision_hidden_size": model_config.get("vision_hidden_size", 1024),
            "image_size": model_config.get("image_size", 448),
            "projector_type": model_config.get("projector_type", "mlp"),
            "llm_type": llm_type,
            "llm_model_name": llm_model_name,
            "torch_dtype": model_config.get("torch_dtype", "float16"),
        })
        
        # Try to load saved weights
        model_weights_path = Path(model_path) / "pytorch_model.bin"
        if model_weights_path.exists():
            model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    
    model.to(device)
    model.eval()
    
    # Setup processor
    processor = VinternProcessor(
        image_size=model_config.get("image_size", 448),
        max_tiles=config.get("data", {}).get("max_tiles", 6),
    )
    
    return model, tokenizer, processor


def generate_answer(
    model: VLMModel,
    image: Image.Image,
    question: str,
    tokenizer: AutoTokenizer,
    processor: VinternProcessor,
    max_new_tokens: int = 256,
    device: str = "cuda",
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate answer for a single image and question
    """
    model.eval()
    
    # Process image
    image_data = processor.preprocess(image)
    pixel_values = image_data["pixel_values"].unsqueeze(0).to(device)  # (1, N, 3, H, W)
    
    # Format prompt (Qwen2 chat format)
    prompt = (
        "<|im_start|>system\n"
        "Bạn là một trợ lý AI hữu ích có khả năng phân tích hình ảnh.<|im_end|>\n"
        "<|im_start|>user\n"
        "<image>\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    
    # Decode
    generated_text = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],  # Skip prompt tokens
        skip_special_tokens=True,
    )
    
    # Clean up: remove end tokens
    for end_token in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        if end_token in generated_text:
            generated_text = generated_text.split(end_token)[0]
    
    return generated_text.strip()


def parse_eval_item(item: Dict) -> tuple:
    """
    Parse evaluation item to extract question and answers
    
    Args:
        item: Data item from eval JSON
        
    Returns:
        (question, answers) where answers is a list for multi-reference support
    """
    question = None
    answers = []
    
    if "conversations" in item:
        # Multi-turn conversation format
        for turn in item["conversations"]:
            role = turn.get("role") or turn.get("from", "")
            content = turn.get("content") or turn.get("value", "")
            
            if role in ["user", "human"] and not question:
                # Remove image token from question
                question = content.replace("<image>\n", "").replace("<image>", "").strip()
            elif role in ["assistant", "gpt"]:
                answers.append(content.strip())
    
    # Fallback to simple format
    if not question:
        question = item.get("question", item.get("prompt", "What is in this image?"))
    
    if not answers:
        # Support multiple answer formats
        if "answers" in item:
            answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
        elif "answer" in item:
            answers = [item["answer"]] if isinstance(item["answer"], str) else item["answer"]
        else:
            answers = [""]
    
    return question, answers


def compute_metrics_full(
    predictions: List[str],
    references: List[List[str]],
    metrics_to_compute: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute full metrics using metrics/ module
    
    Args:
        predictions: List of model predictions
        references: List of reference answers (list of lists for multi-ref)
        metrics_to_compute: List of metrics to compute, None for all
        
    Returns:
        Dictionary with metric scores
    """
    if not FULL_METRICS_AVAILABLE:
        print("Full metrics not available. Using basic exact match.")
        # Basic exact match
        scores = []
        for pred, refs in zip(predictions, references):
            pred_lower = pred.lower().strip()
            match = any(pred_lower == ref.lower().strip() for ref in refs)
            scores.append(1.0 if match else 0.0)
        return {"exact_match": np.mean(scores)}
    
    # Use compute_all_data from metrics/compute_score.py
    results = compute_all_data(references, predictions)
    
    # Extract average scores
    metrics = {}
    for metric_name, metric_data in results.items():
        if isinstance(metric_data, dict):
            metrics[metric_name] = metric_data.get("average", 0.0)
        else:
            metrics[metric_name] = float(metric_data)
    
    return metrics


def evaluate(
    model_path: str,
    eval_data_path: str,
    image_dir: str,
    output_dir: str = "eval_results",
    config_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    metrics_list: Optional[List[str]] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    device: str = "cuda",
    save_individual_scores: bool = True,
) -> Dict:
    """
    Evaluate model on test set with full metrics
    
    Args:
        model_path: Path to model checkpoint
        eval_data_path: Path to evaluation data JSON
        image_dir: Directory containing images
        output_dir: Output directory for results
        config_path: Optional config file path
        max_samples: Maximum samples to evaluate (None for all)
        metrics_list: List of metrics to compute (None for all)
        temperature: Generation temperature
        max_new_tokens: Maximum new tokens to generate
        device: Device to run on
        save_individual_scores: Save per-sample scores
        
    Returns:
        Dictionary with metric scores
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model, tokenizer, processor
    print("=" * 50)
    print("Loading Model")
    print("=" * 50)
    model, tokenizer, processor = load_model_and_tokenizer(
        model_path, config_path, device
    )
    
    # Load evaluation data
    print(f"\nLoading evaluation data from: {eval_data_path}")
    with open(eval_data_path) as f:
        eval_data = json.load(f)
    
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    print(f"Evaluating on {len(eval_data)} samples...")
    
    # Generate predictions
    print("\n" + "=" * 50)
    print("Generating Predictions")
    print("=" * 50)
    
    predictions = []
    all_references = []
    results = []
    
    for idx, item in enumerate(tqdm(eval_data, desc="Generating")):
        # Load image
        image_path = Path(image_dir) / item["image"]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            predictions.append("")
            all_references.append([""])
            continue
        
        # Parse question and answers
        question, answers = parse_eval_item(item)
        
        # Generate answer
        try:
            prediction = generate_answer(
                model, image, question, tokenizer, processor,
                max_new_tokens=max_new_tokens,
                device=device,
                temperature=temperature,
            )
        except Exception as e:
            print(f"Error generating answer for sample {idx}: {e}")
            prediction = ""
        
        predictions.append(prediction)
        all_references.append(answers)
        
        # Save result
        result = {
            "id": idx,
            "image": str(item["image"]),
            "question": question,
            "references": answers,
            "prediction": prediction,
        }
        results.append(result)
    
    # Compute metrics
    print("\n" + "=" * 50)
    print("Computing Metrics")
    print("=" * 50)
    
    metric_scores = compute_metrics_full(predictions, all_references, metrics_list)
    
    # Add individual scores if requested
    if save_individual_scores and FULL_METRICS_AVAILABLE:
        # Compute per-sample scores
        calculator = ScoreCalculator()
        for i, (pred, refs) in enumerate(zip(predictions, all_references)):
            if i < len(results):
                try:
                    results[i]["scores"] = {
                        "accuracy": calculator.accuracy_score(refs, pred),
                        "f1_token": calculator.f1_token(refs, pred),
                        "bleu": calculator.bleu_score(refs, pred),
                    }
                except:
                    results[i]["scores"] = {}
    
    # Save results
    print("\nSaving results...")
    
    # Save predictions
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metric_scores, f, indent=2)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    for metric, score in sorted(metric_scores.items()):
        print(f"  {metric:15s}: {score:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return metric_scores


def evaluate_with_gpt4(
    results_path: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Evaluate predictions using GPT-4o
    
    Args:
        results_path: Path to predictions.json from evaluate()
        api_key: OpenAI API key
        model: GPT model to use
        max_samples: Max samples to evaluate (for cost control)
    
    Returns:
        Scoring results
    """
    try:
        import openai
    except ImportError:
        print("OpenAI package not installed. Install with: pip install openai")
        return {}
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return {}
    
    client = openai.OpenAI(api_key=api_key)
    
    # Load results
    with open(results_path) as f:
        results = json.load(f)
    
    if max_samples:
        results = results[:max_samples]
    
    # Evaluation prompt
    evaluation_prompt = """Đánh giá câu trả lời sau cho một câu hỏi về hình ảnh trên thang điểm 0-10:

Câu hỏi: {question}
Đáp án tham khảo: {references}
Câu trả lời của model: {prediction}

Hãy cho điểm từ 0 đến 10 dựa trên:
- Độ chính xác so với đáp án tham khảo
- Tính đầy đủ của thông tin
- Độ mạch lạc của câu trả lời

Format:
SCORE: <số điểm>
REASONING: <giải thích ngắn gọn>"""
    
    scores = []
    detailed_results = []
    
    print(f"\nEvaluating {len(results)} predictions with GPT-4...")
    for item in tqdm(results):
        refs = item.get("references", [item.get("reference", "")])
        if isinstance(refs, str):
            refs = [refs]
        
        prompt = evaluation_prompt.format(
            question=item["question"],
            references=" | ".join(refs),
            prediction=item["prediction"],
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            
            text = response.choices[0].message.content
            
            # Extract score
            score = None
            if "SCORE:" in text:
                score_str = text.split("SCORE:")[1].split("\n")[0].strip()
                try:
                    score = float(score_str.split()[0])
                    scores.append(score)
                except ValueError:
                    pass
            
            detailed_results.append({
                "id": item.get("id"),
                "gpt4_score": score,
                "gpt4_response": text,
            })
            
        except Exception as e:
            print(f"Error evaluating item: {e}")
            detailed_results.append({
                "id": item.get("id"),
                "gpt4_score": None,
                "error": str(e),
            })
    
    # Compute statistics
    valid_scores = [s for s in scores if s is not None]
    
    results_dict = {
        "gpt4_avg_score": np.mean(valid_scores) if valid_scores else 0.0,
        "gpt4_std_score": np.std(valid_scores) if valid_scores else 0.0,
        "gpt4_min_score": min(valid_scores) if valid_scores else 0.0,
        "gpt4_max_score": max(valid_scores) if valid_scores else 0.0,
        "gpt4_n_samples": len(valid_scores),
        "gpt4_model": model,
    }
    
    print(f"\nGPT-4 Evaluation Results:")
    print(f"  Average Score: {results_dict['gpt4_avg_score']:.2f}/10")
    print(f"  Std Dev: {results_dict['gpt4_std_score']:.2f}")
    print(f"  Range: [{results_dict['gpt4_min_score']:.1f}, {results_dict['gpt4_max_score']:.1f}]")
    
    return results_dict, detailed_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM Model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--eval_data", type=str, required=True, 
                        help="Path to evaluation data JSON")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing images")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory for results")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    
    # GPT-4 evaluation
    parser.add_argument("--use_gpt4", action="store_true",
                        help="Use GPT-4 for evaluation")
    parser.add_argument("--gpt4_model", type=str, default="gpt-4o-mini",
                        help="GPT-4 model to use")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key")
    parser.add_argument("--gpt4_max_samples", type=int, default=None,
                        help="Max samples for GPT-4 eval (cost control)")
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    
    # GPT-4 evaluation if requested
    if args.use_gpt4:
        predictions_path = Path(args.output_dir) / "predictions.json"
        gpt4_metrics, gpt4_details = evaluate_with_gpt4(
            str(predictions_path),
            api_key=args.api_key,
            model=args.gpt4_model,
            max_samples=args.gpt4_max_samples,
        )
        
        # Merge metrics
        metrics.update(gpt4_metrics)
        
        # Save updated metrics
        with open(Path(args.output_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save GPT-4 detailed results
        with open(Path(args.output_dir) / "gpt4_details.json", "w", encoding="utf-8") as f:
            json.dump(gpt4_details, f, ensure_ascii=False, indent=2)
    
    print("\n✓ Evaluation completed!")
    return metrics


if __name__ == "__main__":
    main()
