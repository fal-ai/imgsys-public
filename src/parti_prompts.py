import time
import os
import datasets
import asyncio
import random
import fal_client
from collections import Counter
from supabase import create_client

SEMAPHORE = asyncio.Semaphore(48)
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
        "name": "SDXL-Lightning (2 steps)",
        "type": "fal",
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning",
        "fal": {
            "model_id": "fal-ai/fast-lightning-sdxl",
            "playground": "https://fal.ai/models/stable-diffusion-xl-lightning",
            "params": {
                "num_inference_steps": 2,
            },
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
        "name": "Hyper SD - SDXL (1 steps)",
        "type": "fal",
        "url": "https://huggingface.co/ByteDance/Hyper-SD",
        "fal": {
            "model_id": "fal-ai/hyper-sdxl",
            "playground": "https://fal.ai/models/fal-ai/hyper-sdxl",
            "params": {
                "num_inference_steps": 1,
            },
        },
    },
    {
        "name": "Hyper SD - SDXL (2 steps)",
        "type": "fal",
        "url": "https://huggingface.co/ByteDance/Hyper-SD",
        "fal": {
            "model_id": "fal-ai/hyper-sdxl",
            "playground": "https://fal.ai/models/fal-ai/hyper-sdxl",
            "params": {
                "num_inference_steps": 2,
            },
        },
    },
    {
        "name": "Hyper SD - SDXL (4 steps)",
        "type": "fal",
        "url": "https://huggingface.co/ByteDance/Hyper-SD",
        "fal": {
            "model_id": "fal-ai/hyper-sdxl",
            "playground": "https://fal.ai/models/fal-ai/hyper-sdxl",
            "params": {
                "num_inference_steps": 4,
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
        "url": "https://huggingface.co/dataautogpt3/ProteusV0.3",
        "fal": {
            "model_id": "fal-ai/any-sd",
            "playground": "https://fal.ai/models/any-stable-diffusion-xl",
            "params": {
                "model_name": "dataautogpt3/ProteusV0.3",
                "image_size": "square_hd",
                "num_inference_steps": 40,
                "guidance_scale": 7.5,
            },
        },
    },
    {
        "name": "Kandinsky",
        "type": "fal",
        "url": "https://huggingface.co/kandinsky-community/kandinsky-3",
        "fal": {
            "model_id": "fal-ai/kandinsky3",
            "playground": "https://fal.ai/models/kandinsky-3",
            "params": {
                "num_inference_steps": 35,
            },
        },
    },
    {
        "name": "Dreamshaper v8",
        "type": "fal",
        "url": "https://huggingface.co/Lykon/dreamshaper-8",
        "fal": {
            "model_id": "fal-ai/dreamshaper",
            "playground": "https://fal.ai/models/dreamshaper",
            "params": {
                "model_name": "Lykon/dreamshaper-8",
                "num_inference_steps": 35,
                "image_size": {
                    "width": 768,
                    "height": 768,
                },
                "guidance_scale": 7.5,
            },
        },
    },
    {
        "name": "Dreamshaper SDXL-1-0",
        "type": "fal",
        "url": "https://huggingface.co/Lykon/dreamshaper-xl-1-0",
        "fal": {
            "model_id": "fal-ai/dreamshaper",
            "playground": "https://fal.ai/models/dreamshaper",
            "params": {
                "model_name": "Lykon/dreamshaper-xl-1-0",
                "num_inference_steps": 35,
                "image_size": "square_hd",
            },
            "guidance_scale": 7.5,
        },
    },
    {
        "name": "Realistic Vision V6.0",
        "type": "fal",
        "url": "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "fal": {
            "model_id": "fal-ai/realistic-vision",
            "playground": "https://fal.ai/models/realistic-vision",
            "params": {
                "model_name": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
                "image_size": {
                    "width": 768,
                    "height": 768,
                },
                "guidance_scale": 7.5,
                "num_inference_steps": 35,
            },
        },
    },
    {
        "name": "RealVisXL V4.0",
        "type": "fal",
        "url": "https://huggingface.co/SG161222/RealVisXL_V4.0",
        "fal": {
            "model_id": "fal-ai/realistic-vision",
            "playground": "https://fal.ai/models/realistic-vision",
            "params": {
                "model_name": "SG161222/RealVisXL_V4.0",
                "guidance_scale": 7.5,
                "num_inference_steps": 35,
                "image_size": "square_hd",
            },
        },
    },
    {
        "name": "Mobius",
        "type": "fal",
        "url": "https://huggingface.co/Corcelio/mobius",
        "fal": {
            "model_id": "fal-ai/any-sd",
            "playground": "https://fal.ai/models/mobius",
            "params": {
                "scheduler": "KDPM 2A",
                "model_name": "Corcelio/mobius",
                "guidance_scale": 7,
                "negative_prompt": "",
                "num_inference_steps": 50,
                "clip_skip": 3,
                "image_size": "square_hd",
            },
        },
    },
    {
        "name": "sweet-nothings-1947",
        "type": "fal",
        "url": "https://huggingface.co/404",
        "fal": {
            "model_id": "fal-ai/any-sd",
            "playground": "https://fal.ai/models/any-stable-diffusion-xl",
            "params": {
                "scheduler": "Euler",
                "model_name": "anonymous-t2i-model/sweet-nothings-1947",
                "guidance_scale": 7,
                "negative_prompt": "",
                "num_inference_steps": 35,
                "image_size": "square_hd",
            },
        },
    },
    {
        "name": "dpo-sdxl-text2image-v1",
        "type": "fal",
        "url": "https://huggingface.co/mhdang/dpo-sdxl-text2image-v1",
        "fal": {
            "model_id": "fal-ai/any-sd",
            "playground": "https://fal.ai/models/any-stable-diffusion-xl",
            "params": {
                "scheduler": "Euler",
                "model_name": "sayakpaul/dpo-sdxl-text2image-v1-full",
                "guidance_scale": 7,
                "negative_prompt": "",
                "num_inference_steps": 35,
                "image_size": "square_hd",
            },
        },
    },
    {
        "name": "ColorfulXL-Lightning",
        "type": "fal",
        "url": "https://huggingface.co/recoilme/ColorfulXL-Lightning",
        "fal": {
            "model_id": "fal-ai/any-sd",
            "playground": "https://fal.ai/models/any-stable-diffusion-xl",
            "params": {
                "model_name": "recoilme/ColorfulXL-Lightning",
                "image_size": "square_hd",
                "scheduler": "Euler A",
                "guidance_scale": 1.5,
                "num_inference_steps": 9,
            },
        },
    },
    {
        "name": "FluentlyXL v4",
        "type": "fal",
        "url": "https://huggingface.co/fluently/Fluently-XL-v4",
        "fal": {
            "model_id": "fal-ai/any-sd",
            "playground": "https://fal.ai/models/any-stable-diffusion-xl",
            "params": {
                "model_name": "fluently/Fluently-XL-v4",
                "image_size": "square_hd",
                "guidance_scale": 5.5,
                "num_inference_steps": 30,
                "scheduler": "Euler A",
            },
        },
    },
]

IGNORED_WORDS = [
    "girl",
    "woman",
    "princess",
    "queen",
    "fairy",
    "angel",
    "lady",
    "child",
    "goddess",
    "female",
    "nsfw",
    "unsafe",
    "sexy",
    "beautiful",
    "pretty",
    "hot",
    "nude",
    "erotic",
    "putin",
    "war",
    "kill",
    "trump",
    "biden",
    "dead",
    "demon",
    "satan",
    "god",
    "jesus",
    "kissing",
    "lovers",
    "pee",
    "poop",
    "vomit",
    "blood",
    "gore",
    "creepy",
    "scary",
    "clown",
    "joker",
    "baby",
]


async def run_model(model, prompt):
    params = model["fal"].get("params", {})
    return await fal_client.run_async(
        model["fal"]["model_id"],
        arguments={
            "prompt": prompt,
            **params,
            "enable_safety_checker": True,
            "sync_mode": False,
            "format": "png",
            "safety_checker_version": "v2",
        },
        hint=params.get("model_name"),
    )


async def prepare_sample(prompt, supabase_client, model_weights):
    # Choose model_a with weight and model_b randomly. This makes sure that the newly
    # added models are still competing with the old/sampled models.
    (model_a,) = random.choices(MODEL_MAP, weights=model_weights, k=1)
    model_b = random.choice(MODEL_MAP)
    while model_a == model_b:
        model_b = random.choice(MODEL_MAP)

    models = [model_a, model_b]
    random.shuffle(models)
    model_a, model_b = models
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

    image_a = result_a["images"][0]["url"]
    image_b = result_b["images"][0]["url"]
    if image_a.startswith("data:") or image_b.startswith("data:"):
        print("Data URL")
        return

    return {
        "prompt": prompt,
        "model_a": model_a["name"],
        "model_b": model_b["name"],
        "image_a": result_a["images"][0]["url"],
        "image_b": result_b["images"][0]["url"],
    }


async def main():
    dataset = datasets.load_dataset(
        "isidentical/random-stable-diffusion-prompts",
        split="train",
        keep_in_memory=True,
        revision="b63bf47a491dac6a8f8bfe8f949b10baff73e680",
    )
    supabase_client = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY"),
    )

    model_counts = Counter()
    result = (
        supabase_client.table("latest_ratings").select("*", count="exact").execute()
    )
    for row in result.data:
        model_counts[row["model_name"]] = row["num_samples"]

    max_count = max(model_counts.values())
    model_weights = [
        max_count - model_counts.get(model["name"], 0) + 1 for model in MODEL_MAP
    ]

    for _ in range(50):
        t0 = time.perf_counter()
        dataset = dataset.shuffle()
        futures: list[asyncio.Future] = []
        for prompt in dataset[: 48 * 4]["prompt"]:
            futures.append(
                asyncio.create_task(
                    prepare_sample(prompt.strip(), supabase_client, model_weights)
                )
            )
        t1 = time.perf_counter()
        print("Time to prepare samples", t1 - t0)

        t0 = time.perf_counter()
        results = []
        try:
            for n, future in enumerate(asyncio.as_completed(futures, timeout=180), 1):
                try:
                    sample = await future
                except Exception:
                    import traceback

                    traceback.print_exc()
                    print("Error!!!")
                    continue

                if sample:
                    results.append(sample)
                    print(f"{n}/{len(futures)}", sample)
        except asyncio.TimeoutError:
            print("Timeout!!!")

        if not results:
            break
        print("img/sec", len(results) / (time.perf_counter() - t0))
        supabase_client.table("parti_prompts").insert(results).execute()
        print("============ CYCLE OVER ========")
        for future in futures:
            if future.done():
                continue
            future.cancel()

        await asyncio.sleep(2)
        for future in futures:
            try:
                await future
            except asyncio.CancelledError:
                pass
            except Exception:
                import traceback

                traceback.print_exc()
                print("Error!!!")
                continue


if __name__ == "__main__":
    asyncio.run(main())
