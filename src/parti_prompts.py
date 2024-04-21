import os
import datasets
import asyncio
import random
import fal_client
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
]


async def run_model(model, prompt):
    return await fal_client.run_async(
        model["fal"]["model_id"],
        arguments={
            "prompt": prompt,
            **model["fal"].get("params", {}),
            "enable_safety_checker": True,
            "sync_mode": False,
        },
    )


async def prepare_sample(prompt, supabase_client):
    model_a, model_b = random.sample(MODEL_MAP, 2)
    async with SEMAPHORE:
        try:
            result_a, result_b = await asyncio.gather(
                run_model(model_a, prompt),
                run_model(model_b, prompt),
            )
        except Exception:
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

    for _ in range(50):
        dataset = dataset.shuffle()
        futures = []
        for prompt in dataset[:32]["Prompt"]:
            futures.append(prepare_sample(prompt, supabase_client))

        results = []
        for future in asyncio.as_completed(futures):
            sample = await future
            if sample:
                results.append(sample)
                print(sample)

        supabase_client.table("parti_prompts").insert(results).execute()


if __name__ == "__main__":
    asyncio.run(main())
