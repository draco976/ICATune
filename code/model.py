import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model


def create_llama_model(model_path="model/llama3", quantize=True, device="cuda"):
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


def create_mistral_model(model_path="mistralai/Mistral-7B-v0.3", quantize=True, device="cuda"):
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
    if layers is None or '32' in layers:
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


def apply_lora(model, layers=None, r=16, alpha=8, dropout=0.05):
    """
    Apply LoRA to a model.
    
    Args:
        model: Model to apply LoRA to
        layers: Specific layers to apply LoRA to
        r: Rank of low-rank matrices
        alpha: Scaling factor
        dropout: Dropout probability
        
    Returns:
        Model with LoRA applied
    """
    # Convert layers string to list if needed
    if isinstance(layers, str):
        layers = layers.split('-')
        
    lora_config = setup_lora_config(layers, r, alpha, dropout)
    model = get_peft_model(model, lora_config)
    
    return model