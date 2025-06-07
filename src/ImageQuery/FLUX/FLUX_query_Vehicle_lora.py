import os
from diffusers import FluxPipeline
import torch

def main(base_model="black-forest-labs/FLUX.1-dev", weight_name="pytorch_lora_weights.safetensors"):
    models_dir = "xxx"
    
    prompts = [
        "An image of a bicycle and a rider.",
        # Remove a Noun
        "An image of a bicycle.",
        "An image of a rider.",
        # Switch Nouns
        "An image of a motorcycle and a rider.",
        "An image of a scooter and a rider.",
        "An image of a horse and a rider.",
        "An image of a bicycle and a driver.",
        "An image of a bicycle and a pedestrian.",
        "An image of a bicycle and a mountain.",
        # Switch Prepositions
        "An image of a bicycle with a rider.",
        "An image of a bicycle near a rider.",
        # Switch Verbs
        "An image of a bicycle being ridden by a rider.",
        "An image of a bicycle parked by a rider.",
        # Add Adjectives
        "An image of a red bicycle and a rider.",
        "An image of a bicycle and a tall rider."
    ]

    output_dir = "xxx"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {base_model}")
    pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    model_names = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]

    for model_name in model_names:
        if "bicycle" in model_name or "Bicycle" in model_name:
            print(f"Processing LoRA Weights: {model_name}")
            lora_path = os.path.join(models_dir, model_name)

            pipe.load_lora_weights(lora_path, weight_name=weight_name)

            for prompt in prompts:
                print(f"Prompt: {prompt} for LoRA Weight: {model_name}")

                prompt_folder_name = prompt.replace(' ', '_').replace('.', '')
                prompt_output_dir = os.path.join(output_dir, model_name, prompt_folder_name)
                os.makedirs(prompt_output_dir, exist_ok=True)

                for i in range(10):
                    image_filename = f"{model_name}_{prompt_folder_name}_{i + 1}.png"

                    if not os.path.exists(os.path.join(prompt_output_dir, image_filename)):
                        image = pipe(prompt, num_inference_steps=25, guidance_scale=5.0).images[0]
                        image.save(os.path.join(prompt_output_dir, image_filename))

                print(f"Completed generation for prompt: '{prompt}' in model: '{model_name}'")
            pipe.unload_lora_weights()

    print("Done!")

if __name__ == '__main__':
    main()
