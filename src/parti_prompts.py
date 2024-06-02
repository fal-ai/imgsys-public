import os
import datasets
import asyncio
import random
import fal_client
from collections import Counter
from supabase import create_client

SEMAPHORE = asyncio.Semaphore(16)
MODEL_MAP = [
    {
        "name": "Stable Diffusion XL 1.0 (Base)",
        "type": "fal",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "fal": {
            "model_id": "fal-ai/fast-sdxl",
            "playground": "https://fal.ai/models/stable-diffusion-xl",
            "params": {
                "num_inference_steps": 35,
            },
        },
    },
    {
        "name": "Fooocus (Quality)",
        "type": "fal",
        "url": "https://github.com/lllyasviel/Fooocus",
        "fal": {
            "model_id": "fal-ai/fooocus",
            "playground": "https://fal.ai/models/fooocus",
            "params": {
                "performance": "Quality",
                "output_format": "png",
            },
        },
    },
    {
        "name": "PixArt-Σ",
        "type": "fal",
        "url": "https://pixart-alpha.github.io/PixArt-sigma-project",
        "fal": {
            "model_id": "fal-ai/pixart-sigma",
            "playground": "https://fal.ai/models/pixart-sigma",
        },
    },
    {
        "name": "SDXL Turbo",
        "type": "fal",
        "url": "https://huggingface.co/stabilityai/sdxl-turbo",
        "fal": {
            "model_id": "fal-ai/fast-turbo-diffusion",
            "playground": "https://fal.ai/models/fast-turbo-diffusion-turbo",
            "params": {
                "num_inference_steps": 4,
                "guidance_scale": 0,
            },
        },
    },
    {
        "name": "SD Turbo",
        "type": "fal",
        "url": "https://huggingface.co/stabilityai/sd-turbo",
        "fal": {
            "model_id": "fal-ai/fast-turbo-diffusion",
            "playground": "https://fal.ai/models/fast-turbo-diffusion-turbo",
            "params": {
                "model_name": "stabilityai/sd-turbo",
                "num_inference_steps": 4,
                "guidance_scale": 0,
            },
        },
    },
    {
        "name": "Latent Consistency Model (LCM): SDXL",
        "type": "fal",
        "url": "https://huggingface.co/latent-consistency/lcm-sdxl",
        "fal": {
            "model_id": "fal-ai/fast-lcm-diffusion",
            "playground": "https://fal.ai/models/fast-lcm-diffusion-turbo",
            "params": {
                "num_inference_steps": 8,
            },
        },
    },
    {
        "name": "Playground v2.5 (Aesthetic Model)",
        "type": "fal",
        "url": "https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic",
        "fal": {
            "model_id": "fal-ai/playground-v25",
            "playground": "https://fal.ai/models/playground-v25",
        },
    },
    {
        "name": "Stable Cascade",
        "type": "fal",
        "url": "https://huggingface.co/stabilityai/stable-cascade",
        "fal": {
            "model_id": "fal-ai/stable-cascade",
            "playground": "https://fal.ai/models/stable-cascade",
        },
    },
    {
        "name": "SDXL-Lightning (4 steps)",
        "type": "fal",
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning",
        "fal": {
            "model_id": "fal-ai/fast-lightning-sdxl",
            "playground": "https://fal.ai/models/stable-diffusion-xl-lightning",
            "params": {
                "num_inference_steps": 4,
            },
        },
    },
    {
        "name": "SDXL-Lightning (8 steps)",
        "type": "fal",
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning",
        "fal": {
            "model_id": "fal-ai/fast-lightning-sdxl",
            "playground": "https://fal.ai/models/stable-diffusion-xl-lightning",
            "params": {
                "num_inference_steps": 8,
            },
        },
    },
    {
        "name": "Stable Diffusion v1.5",
        "type": "fal",
        "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5",
        "fal": {
            "model_id": "fal-ai/stable-diffusion-v15",
            "playground": "https://fal.ai/models/stable-diffusion-v1.5",
            "params": {
                "image_size": {
                    "width": 512,
                    "height": 512,
                },
                "num_inference_steps": 35,
            },
        },
    },
    {
        "name": "Fooocus (Refined SDXL LCM)",
        "type": "fal",
        "url": "https://fal.ai/models/fooocus-extreme-speed",
        "fal": {
            "model_id": "fal-ai/fast-fooocus-sdxl",
            "playground": "https://fal.ai/models/fooocus-extreme-speed",
        },
    },
    {
        "name": "Juggernaut XL v9",
        "type": "fal",
        "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9",
        "fal": {
            "model_id": "fal-ai/any-sd",
            "playground": "https://fal.ai/models/any-stable-diffusion-xl",
            "params": {
                "model_name": "RunDiffusion/Juggernaut-XL-v9",
                "image_size": "square_hd",
            },
        },
    },
    {
        "name": "Proteus",
        "type": "fal",
        "url": "https://huggingface.co/dataautogpt3/Proteus-RunDiffusion",
        "fal": {
            "model_id": "fal-ai/any-sd",
            "playground": "https://fal.ai/models/any-stable-diffusion-xl",
            "params": {
                "model_name": "dataautogpt3/Proteus-RunDiffusion",
                "image_size": "square_hd",
                "guidance_scale": 8.5,
            },
        },
    },
    {
        "name": "Deliberate v6",
        "type": "fal",
        "url": "https://huggingface.co/XpucT/Deliberate",
        "fal": {
            "model_id": "XpucT/Deliberate",
            "playground": "https://fal.ai/models/fal-ai/lora",
            "params": {
                "image_size": {
                    "width": 512,
                    "height": 512,
                },
                "num_inference_steps": 30,
                "guidance_scale": 6,
            },
        },
        
    },
    {
        "name": "Reliberate v3",
        "type": "fal",
        "url": "https://huggingface.co/XpucT/Reliberate",
        "fal": {
            "model_id": "XpucT/Reliberate",
            "playground": "https://fal.ai/models/fal-ai/lora",
            "params": {
                "image_size": {
                    "width": 512,
                    "height": 512,
                },
                "num_inference_steps": 30,
                "guidance_scale": 6,
            },
        },
        
    },
]


async def run_model(model, prompt):
    return await fal_client.run_async(
        model["fal"]["model_id"],
        arguments={
            "prompt": prompt,
            **model["fal"].get("params", {}),
            "enable_safety_checker": True,
            "sync_mode": False,
            "format": "png",
        },
    )


async def prepare_sample(prompt, supabase_client, model_weights):
    # Choose model_a with weight and model_b randomly, but not the same
    (model_a,) = random.choices(MODEL_MAP, weights=model_weights, k=1)
    model_b = random.choice(MODEL_MAP)
    while model_a == model_b:
        model_b = random.choice(MODEL_MAP)
    async with SEMAPHORE:
        try:
            result_a, result_b = await asyncio.gather(
                run_model(model_a, prompt),
                run_model(model_b, prompt),
            )
        except Exception:
            import traceback

            traceback.print_exc()
            print("Error!!!")
            return

    has_nsfw_concepts_a = result_a.get("has_nsfw_concepts", [False])
    has_nsfw_concepts_b = result_b.get("has_nsfw_concepts", [False])
    if has_nsfw_concepts_a[0] or has_nsfw_concepts_b[0]:
        print("NSFW")
        return

    return {
        "prompt": prompt,
        "model_a": model_a["name"],
        "model_b": model_b["name"],
        "image_a": result_a["images"][0]["url"],
        "image_b": result_b["images"][0]["url"],
    }


async def main():
    dataset = datasets.load_dataset("nateraw/parti-prompts", split="train")
    supabase_client = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY"),
    )

    model_counts = Counter()
    for model in MODEL_MAP:
        for position in ["model_a", "model_b"]:
            count = (
                supabase_client.table("parti_prompts")
                .select("*", count="exact")
                .eq(position, model["name"])
                .execute()
            ).count
            model_counts[model["name"]] += count

    max_count = max(model_counts.values())
    model_weights = [
        max_count - model_counts.get(model["name"], 0) + 1 for model in MODEL_MAP
    ]
    print(model_weights)

    for _ in range(50):
        dataset = dataset.shuffle()
        futures = []
        for prompt in dataset[:32]["Prompt"]:
            futures.append(prepare_sample(prompt, supabase_client, model_weights))

        results = []
        for future in asyncio.as_completed(futures):
            sample = await future
            if sample:
                results.append(sample)
                print(sample)

        if not results:
            break
        supabase_client.table("parti_prompts").insert(results).execute()


if __name__ == "__main__":
    asyncio.run(main())
