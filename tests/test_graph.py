import unittest

from helpers import registry

from jena_image_render.config import expand
from jena_image_render.graph import OUTPUT_NODE, Names, build_graph


def render(model, **job):
    data = {"version": 1, "model": model, "prompt": "p", **job}
    return expand(data, registry(), "t")[0]


def classes(graph):
    return {key: node["class_type"] for key, node in graph.items()}


NAMES = Names("base.safetensors", "te.safetensors", "vae.safetensors")


class GraphTests(unittest.TestCase):
    def test_every_engine_ends_in_preview(self) -> None:
        for model in ("zimage", "krea2", "qwen", "klein", "ill"):
            with self.subTest(model=model):
                graph = build_graph(render(model), NAMES)
                self.assertEqual(graph[OUTPUT_NODE]["class_type"], "PreviewImage")
                self.assertEqual(graph["decode"]["class_type"], "VAEDecode")

    def test_zimage_template(self) -> None:
        graph = build_graph(render("zimage"), NAMES)
        self.assertEqual(graph["clip"]["inputs"]["type"], "lumina2")
        self.assertEqual(graph["shift"]["inputs"]["shift"], 3.0)
        self.assertEqual(classes(graph)["negative"], "ConditioningZeroOut")
        self.assertEqual(classes(graph)["latent"], "EmptySD3LatentImage")
        self.assertEqual(graph["sample"]["inputs"]["cfg"], 1.0)

    def test_krea2_guidance_is_cfg_minus_one(self) -> None:
        graph = build_graph(render("krea2", guidance=2.5, negative="blur"), NAMES)
        self.assertEqual(graph["sample"]["inputs"]["cfg"], 3.5)
        self.assertEqual(classes(graph)["negative"], "CLIPTextEncode")
        turbo = build_graph(render("krea2"), NAMES)
        self.assertEqual(classes(turbo)["negative"], "ConditioningZeroOut")

    def test_qwen_uses_its_encoder_and_plain_latent(self) -> None:
        graph = build_graph(render("qwen"), NAMES)
        self.assertEqual(classes(graph)["encode"], "TextEncodeQwenImage21")
        self.assertEqual(classes(graph)["cache"], "QwenImage21Cache")
        self.assertEqual(classes(graph)["latent"], "EmptyLatentImage")
        self.assertEqual(graph["sample"]["inputs"]["positive"], ["encode", 0])

    def test_klein_uses_custom_sampler(self) -> None:
        graph = build_graph(render("klein"), NAMES)
        kinds = classes(graph)
        self.assertEqual(kinds["sample"], "SamplerCustomAdvanced")
        self.assertEqual(kinds["latent"], "EmptyFlux2LatentImage")
        self.assertEqual(graph["clip"]["inputs"]["type"], "flux2")

    def test_illustrious_loras_chain_model_and_clip(self) -> None:
        r = render("ill", loras=[{"name": "istyle", "scale": 0.6}])
        graph = build_graph(r, Names("ckpt.safetensors", loras=("illustrious/is.safetensors",)))
        lora = graph["lora0"]["inputs"]
        self.assertEqual(graph["lora0"]["class_type"], "LoraLoader")
        self.assertEqual((lora["strength_model"], lora["strength_clip"]), (0.6, 0.6))
        self.assertEqual(graph["positive"]["inputs"]["clip"], ["lora0", 1])
        self.assertEqual(graph["sample"]["inputs"]["model"], ["lora0", 0])
        self.assertEqual(graph["sample"]["inputs"]["cfg"], 6.0)

    def test_unet_loras_are_model_only(self) -> None:
        r = render("zimage", loras=["zstyle"])
        graph = build_graph(r, Names("b", "c", "v", loras=("zimage/zs.safetensors",)))
        self.assertEqual(graph["lora0"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["shift"]["inputs"]["model"], ["lora0", 0])

    def test_unresolved_loras_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(render("zimage", loras=["zstyle"]), NAMES)


if __name__ == "__main__":
    unittest.main()
