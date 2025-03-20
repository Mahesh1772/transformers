# Vision-Language Model (VLM) Image Analysis Tool

This tool uses LLaVA-1.5, a powerful vision-language model, to analyze images based on text prompts. It can run locally on Windows (or any platform) and is free to use.

## Features

- Analyze any image with natural language prompts
- Get detailed responses about image content
- Supports any query type: object identification, scene description, reasoning about image content
- Fully local execution - no API keys or cloud services required
- Memory optimization options for both CPU and GPU

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers library
- 8GB+ RAM (16GB+ recommended)
- GPU with at least 8GB VRAM (for faster inference) or CPU-only mode

## Setup

1. Install the required packages:

```bash
pip install torch torchvision transformers pillow
```

2. For better GPU memory efficiency, install the optional bitsandbytes library:

```bash
pip install bitsandbytes
```

3. Download the model (it will be automatically downloaded when first running the script, or you can pre-download it):

```bash
# Pre-download the model (optional)
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf'); AutoModelForCausalLM.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype='auto')"
```

## Usage

Run the script with an image and a prompt:

```bash
python qwen_vlm.py --image path/to/your/image.jpg --prompt "What's happening in this image?"
```

### Memory-Optimized Usage

For GPU with limited VRAM (8GB or less):

```bash
python qwen_vlm.py --image image.jpg --prompt "Describe this image" --load_in_4bit
```

For CPU-only environments with limited RAM:

```bash
python qwen_vlm.py --image image.jpg --prompt "Describe this image" --device cpu --use_smaller_model
```

### Command-line Arguments

- `--image`: Path to the image file (required)
- `--prompt`: Text prompt for the image analysis (required)
- `--model_name`: Model name/path (default: "llava-hf/llava-1.5-7b-hf")
- `--device`: Device to run inference on ("cuda" or "cpu", default: CUDA if available, otherwise CPU)
- `--max_length`: Maximum response length (default: 512 tokens)
- `--load_in_4bit`: Load model in 4-bit quantization (saves GPU memory)
- `--use_smaller_model`: Use a smaller model better suited for CPU (BLIP instead of LLaVA)

### Example Prompts

- "Describe what's happening in this image."
- "How many people are in this picture and what are they doing?"
- "What sport is being played in this image?"
- "Identify the main objects in this image."
- "What colors are the team jerseys in this sports scene?"

## Model Information

By default, this tool uses LLaVA-1.5 (7B parameter version), which is:

- A powerful multimodal model fine-tuned on Llama 2
- Capable of understanding and reasoning about images
- Free for personal and research use
- Can run completely locally without an internet connection after downloading

With the `--use_smaller_model` option, it switches to BLIP-VQA, which:
- Is significantly smaller (only a few hundred MB)
- Works well for basic image analysis on CPU
- Has more modest capabilities but is very memory-efficient

## Troubleshooting

- **Out of memory errors on GPU**: Try using `--load_in_4bit` option to reduce memory usage
- **Out of memory errors on CPU**: Use `--use_smaller_model` to switch to a lighter model
- **Slow responses**: GPU acceleration highly recommended for faster inference
- **Model download issues**: Ensure you have a stable internet connection for the initial download

## Alternative Models

You can specify any compatible vision-language model by name:

```bash
# Use LLaVA with Mistral
python qwen_vlm.py --image image.jpg --prompt "Describe this image" --model_name "llava-hf/llava-v1.6-mistral-7b-hf"

# Use BLIP for lighter resource usage
python qwen_vlm.py --image image.jpg --prompt "Describe this image" --model_name "Salesforce/blip-vqa-base"
``` 