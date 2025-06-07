import os
from diffusers import DiffusionPipeline
import torch

def main(base_model="stabilityai/stable-diffusion-xl-base-1.0", weight_name="pytorch_lora_weights.safetensors"):
    models_dir = "xxx"
    
    prompts = [
        "An image of a mouse and a cat.",
        # Remove a noun
        "An image of a mouse.",
        "An image of a cat.",
        # Switch nouns
        "An image of a rat and a cat.",
        "An image of a squirrel and a cat.",
        "An image of a toy and a cat.",
        "An image of a mouse and a dog.",
        "An image of a mouse and a kitten.",
        "An image of a mouse and a sofa.",
        # Switch Prepositions
        "An image of a mouse near a cat.",
        "An image of a mouse behind a cat.",
        # Add Verbs
        "An image of a mouse running from a cat.",
        "An image of a mouse hiding from a cat.",
        # Add Adjectives
        "An image of a small mouse and a cat.",
        "An image of a mouse and a sleeping cat."
    ]

    output_dir = "xxx"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {base_model}")
    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    model_names = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]

    for model_name in model_names:
        if "mice" in model_name or "Mice" in model_name:
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
                        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                        image.save(os.path.join(prompt_output_dir, image_filename))

                print(f"Completed generation for prompt: '{prompt}' in model: '{model_name}'")
            pipe.unload_lora_weights()

    print("Done!")

if __name__ == '__main__':
    main()