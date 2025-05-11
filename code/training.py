import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM, 
    DataCollatorForSeq2Seq, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
import copy
import pickle
import argparse
import numpy as np
import json


class ModelFactory:
    """
    Factory for creating different types of models.
    """
    @staticmethod
    def create_llama(model_path="model/llama3", quantize=True, device="cuda"):
        """
        Create a LLaMA model.
        
        Args:
            model_path: Path to model
            quantize: Whether to apply 4-bit quantization
            device: Device to use
            
        Returns:
            Loaded model
        """
        if quantize:
            model = LlamaForCausalLM.from_pretrained(
                model_path, 
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )
            )
        else:
            model = LlamaForCausalLM.from_pretrained(model_path)
            
        model.eval()
        model.to(device)
        
        return model
    
    @staticmethod
    def create_mistral(model_path="mistralai/Mistral-7B-v0.3", quantize=True, device="cuda"):
        """
        Create a Mistral model.
        
        Args:
            model_path: Path to model
            quantize: Whether to apply 4-bit quantization
            device: Device to use
            
        Returns:
            Loaded model
        """
        if quantize:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
        model.eval()
        model.to(device)
        
        return model


class Preprocessor:
    """
    Text preprocessor for different model types.
    """
    @staticmethod
    def basic_preprocess(example, tokenizer):
        """
        Basic preprocessing for standard models.
        
        Args:
            example: Input example
            tokenizer: Tokenizer to use
            
        Returns:
            Processed example
        """
        example["text"] = example["text"].strip()
        model_inputs = tokenizer(example["text"])  
        model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
        model_inputs["labels"] = [-100 if x > 89 or x < 64 else x for x in model_inputs["labels"]]
        return model_inputs
    
    @staticmethod
    def ft_preprocess(example, tokenizer):
        """
        Preprocessing for fine-tuning with character modification.
        
        Args:
            example: Input example
            tokenizer: Tokenizer to use
            
        Returns:
            Processed example
        """
        example["text"] = ''.join('!' + c if c.isalpha() else c for c in example["text"].strip())
        model_inputs = tokenizer(example["text"])  
        model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
        model_inputs["labels"] = [-100 if x > 89 or x < 64 else x for x in model_inputs["labels"]]
        return model_inputs
    
    
    @staticmethod
    def mistral_preprocess(example, tokenizer):
        """
        Preprocessing for Mistral model.
        
        Args:
            example: Input example
            tokenizer: Tokenizer to use
            
        Returns:
            Processed example
        """
        lower_case_tokens = [x for x in range(97, 123)]
        
        example["text"] = example["text"].strip()
        model_inputs = tokenizer(example["text"])  
        model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
        model_inputs["labels"] = [-100 if x in lower_case_tokens else x for x in model_inputs["labels"]]
        return model_inputs


class MetricsComputer:
    """
    Compute evaluation metrics for different model types.
    """
    @staticmethod
    def compute_basic_metrics(eval_pred, config, tokenizer, pfa_data=None):
        """
        Compute basic accuracy metrics.
        
        Args:
            eval_pred: Evaluation prediction
            examples: Example texts
            pfa_data: PFA data for evaluation
            
        Returns:
            Evaluation metrics
        """

        predictions, labels = eval_pred
        predictions = predictions.argmax(-1)

        total_acc = 0.0
        total_acc_p = np.zeros(c)

        for i in range(len(predictions)):
                indices = np.nonzero(labels[i] != -100)
                label = ''.join(tokenizer.convert_ids_to_tokens(labels[i, indices].squeeze()))
                new_indices = tuple(idx - 1 for idx in indices)
                prediction = ''.join(tokenizer.convert_ids_to_tokens(predictions[i, new_indices].squeeze()))

                acc = 0
                acc_p = np.zeros(config.seq_len)

                psum = 0
                for idx in range(len(config.symbols)):
                        l = config.seq_len
                        if psum == 0:
                                la = label[(-l-psum):]
                                p = prediction[(-l-psum):]
                        else:
                                la = label[(-l-psum):-psum]
                                p = prediction[(-l-psum):-psum]
                        
                        evaluation = pfa_data[i][-(idx+1)].evaluate_sequence(la, p)
                        psum += l
                        acc += np.sum(evaluation)
                        acc_p += evaluation

                acc /= psum
                acc_p /= len(config.symbols)

                total_acc += acc
                total_acc_p += acc_p
            
        return {"accuracy": total_acc / len(predictions)}


class TrainerFactory:
    """
    Factory for creating trainers for different model types.
    """
    @staticmethod
    def create_trainer(model_type, model, tokenizer, dataset_path, 
                      seed=42, learning_rate=2e-4, max_steps=1000, 
                      batch_size=1, save_dir=None):
        """
        Create a trainer for the specified model type.
        
        Args:
            model_type: Type of model (llama, mistral, alignment)
            model: Model to train
            tokenizer: Tokenizer
            dataset_path: Path to dataset
            seed: Random seed
            learning_rate: Learning rate
            max_steps: Maximum number of training steps
            batch_size: Batch size
            save_dir: Directory to save model
            
        Returns:
            Configured trainer
        """
        # Select preprocessor based on model type
        if model_type == "llama":
            preprocess_fn = lambda x: Preprocessor.basic_preprocess(x, tokenizer)
            compute_metrics_fn = MetricsComputer.compute_basic_metrics
        elif model_type == "mistral":
            preprocess_fn = lambda x: Preprocessor.mistral_preprocess(x, tokenizer)
            compute_metrics_fn = MetricsComputer.compute_basic_metrics
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load datasets
        train_dataset = load_dataset("text", data_files=f"{dataset_path}/train.txt").shuffle(seed=seed).map(preprocess_fn)['train']
        val_dataset = load_dataset("text", data_files=f"{dataset_path}/val.txt").map(preprocess_fn)['train']
        config = json.load(open(f"{dataset_path}/config.json", "r"))
        
        # Set up data collator
        data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer)
        
        # Load evaluation data
        pfa_data = None
        examples = None
        try:
            pfa_data = pickle.load(open(f"{dataset_path}/val_pfa_data.pkl", "rb"))
            examples = open(f"{dataset_path}/val.txt", "r").readlines()
        except:
            print("Warning: Could not load PFA data or examples for metrics computation")
            
        # Set up compute metrics function
        compute_metrics_fn_with_data = lambda x: compute_metrics_fn(x, examples, config, tokenizer, pfa_data)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=save_dir or f"model/finetuned/{model_type}",
            overwrite_output_dir=True,
            fp16=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_strategy="steps",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=1000,
            gradient_accumulation_steps=1,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn_with_data,
        )
        
        return trainer


def setup_lora_config(layers=None, r=16, alpha=8, dropout=0.05):
    """
    Set up LoRA configuration.
    
    Args:
        layers: Specific layers to apply LoRA to (None for default)
        r: Rank of low-rank matrices
        alpha: Scaling factor
        dropout: Dropout probability
        
    Returns:
        LoRA configuration
    """
    if layers is None in layers:
        # Apply to all q, k, v projection layers
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
    else:
        # Apply to specific layers
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[f"{layer}.self_attn.q_proj" for layer in layers] + 
                          [f"{layer}.self_attn.k_proj" for layer in layers],
        )


def finetune_model(model_type="llama", dataset="default", layers=None, 
                  seed=42, batch_size=1, learning_rate=2e-4, max_steps=1000,
                  quantize=True, model_path=None, save_dir=None):
    """
    Fine-tune a model.
    
    Args:
        model_type: Type of model (llama, mistral, alignment)
        dataset: Dataset to use
        layers: Layers to apply LoRA to (for LoRA models)
        seed: Random seed
        batch_size: Batch size
        learning_rate: Learning rate
        max_steps: Maximum number of training steps
        quantize: Whether to apply 4-bit quantization
        model_path: Path to model
        save_dir: Directory to save model
        
    Returns:
        Trained model
    """
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    if model_type == "mistral":
        tokenizer = AutoTokenizer.from_pretrained(model_path or "mistralai/Mistral-7B-v0.3")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path or "model/llama3")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    if model_type == "llama":
        model = ModelFactory.create_llama(model_path or "model/llama3", quantize, device)
    elif model_type == "mistral":
        model = ModelFactory.create_mistral(model_path or "mistralai/Mistral-7B-v0.3", quantize, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Parse layers if provided
    if layers is not None and isinstance(layers, str):
        layers = layers.split('-')
        
    lora_config = setup_lora_config(layers)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create trainer
    trainer = TrainerFactory.create_trainer(
        model_type=model_type,
        model=model,
        tokenizer=tokenizer,
        dataset_path=f"data/{dataset}",
        seed=seed,
        learning_rate=learning_rate,
        max_steps=max_steps,
        batch_size=batch_size,
        save_dir=save_dir or f"model/finetuned/{dataset}-{model_type}",
        lora_layers=layers
    )
    
    # Train model
    trainer.train()
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune language models")
    parser.add_argument("--model_type", type=str, default="llama", choices=["llama", "mistral", "alignment"],
                        help="Type of model to fine-tune")
    parser.add_argument("--dataset", type=str, default="monotonic-m3-c2",
                        help="Dataset to use")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layers to apply LoRA to (comma-separated)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of training steps")
    parser.add_argument("--no_quantize", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save model")
    
    args = parser.parse_args()
    
    finetune_model(
        model_type=args.model_type,
        dataset=args.dataset,
        layers=args.layers,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        quantize=not args.no_quantize,
        model_path=args.model_path,
        save_dir=args.save_dir
    )