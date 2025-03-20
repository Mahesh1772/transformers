import os
import sys
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForQuestionAnswering

def setup_parser():
    parser = argparse.ArgumentParser(description="VLM Image Analysis Tool")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--prompt", required=True, help="Prompt for the image analysis")
    parser.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf", help="Model name/path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum response length")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization (saves GPU memory)")
    parser.add_argument("--use_smaller_model", action="store_true", help="Use a smaller model better suited for CPU (BLIP)")
    return parser

def load_model(model_name, device, load_in_4bit=False, use_smaller_model=False):
    """Load the VLM model and processor"""
    if use_smaller_model:
        print("Using smaller BLIP model for CPU compatibility...")
        blip_model_name = "Salesforce/blip-vqa-base"
        
        print(f"Loading {blip_model_name}...")
        # Device info
        if device == "cuda" and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("Using CPU")
            
        try:
            # BLIP uses different processor and model classes
            processor = BlipProcessor.from_pretrained(blip_model_name)
            model = BlipForQuestionAnswering.from_pretrained(
                blip_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # Move model to the right device
            if device == "cuda":
                model = model.to("cuda")
                
            print(f"BLIP model loaded successfully!")
            return processor, model, device, True  # Return True to indicate this is BLIP
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            sys.exit(1)
    else:
        # Load LLaVA or other causal LM
        print(f"Loading {model_name}...")
        
        # Device info
        if device == "cuda" and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("Using CPU")
        
        try:
            # Load processor and model
            processor = AutoProcessor.from_pretrained(model_name)
            
            # Model loading options
            model_kwargs = {
                "low_cpu_mem_usage": True,
            }
            
            # Add appropriate dtype based on device
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
                
                # Add 4-bit quantization if requested and on GPU
                if load_in_4bit:
                    try:
                        import bitsandbytes as bnb
                        print("Using 4-bit quantization to save GPU memory")
                        model_kwargs["load_in_4bit"] = True
                        model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                    except ImportError:
                        print("Warning: bitsandbytes not installed. Cannot use 4-bit quantization.")
                        print("Install with: pip install bitsandbytes")
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            print(f"Model loaded successfully!")
            return processor, model, device, False  # Return False to indicate this is not BLIP
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

def analyze_image(image_path, prompt, processor, model, max_length=512, is_blip=False):
    """Analyze an image with the given prompt using the VLM"""
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Check if we're using BLIP
        if is_blip:
            # BLIP processor/model has a different API
            inputs = processor(image, prompt, return_tensors="pt")
            
            # Move inputs to the appropriate device
            if hasattr(model, "device") and model.device.type == "cuda":
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                
            # Generate the response
            print("Generating response with BLIP...")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_length)
                
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        else:
            # LLaVA format
            # For LLaVA, we format the prompt differently
            prompt_text = f"<image>\n{prompt}"
            
            # Process inputs - LLaVA uses a simpler API
            inputs = processor(prompt_text, image, return_tensors="pt")
            
            # Move inputs to the appropriate device
            if hasattr(model, "device") and torch.device(model.device).type == "cuda":
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            
            # Generate the response
            print("Generating response...")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False
                )
            
            # Decode the response
            response = processor.decode(output[0], skip_special_tokens=True)
            
            # Extract just the model's response after the prompt
            # For LLaVA, the response usually includes the prompt, so we need to extract just the answer
            try:
                if prompt in response:
                    response = response.split(prompt, 1)[1].strip()
                elif "<image>" in response:
                    response = response.split("<image>", 1)[1].strip()
                    if "\n" in response:
                        # The response after the first newline is the actual answer
                        response = response.split("\n", 1)[1].strip()
            except:
                # If parsing fails, return the full response
                pass
                
            return response
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Check if the image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        sys.exit(1)
    
    # Load the model
    processor, model, device, is_blip = load_model(
        args.model_name, 
        args.device, 
        args.load_in_4bit,
        args.use_smaller_model
    )
    
    # Analyze the image
    response = analyze_image(
        args.image, 
        args.prompt, 
        processor, 
        model,
        args.max_length,
        is_blip
    )
    
    # Print the response
    print("\n" + "="*50)
    print("VLM Response:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    main() 