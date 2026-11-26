import unittest

from helpers import registry

from jena_image_render.config import ConfigError, expand, target


def run(data, output="out"):
    return expand(data, registry(), "test.json", default_output=output)


class ConfigTests(unittest.TestCase):
    def test_minimal_single_job_uses_engine_defaults(self) -> None:
        (r,) = run({"version": 1, "model": "zimage", "prompt": "a cafe"})
        self.assertEqual((r.width, r.height, r.steps, r.guidance), (832, 1248, 9, 0.0))
        self.assertEqual((r.sampler, r.scheduler), ("res_multistep", "simple"))
        self.assertEqual(r.output, "out/zimage/job1_0.png")
        self.assertEqual(r.negative, "")

    def test_prompt_x_model_x_seed(self) -> None:
        renders = run(
            {
                "version": 1,
                "output": "cmp",
                "defaults": {"model": ["zimage", "krea2", "qwen"], "seeds": [1, 2]},
                "jobs": [{"name": "a", "prompt": "x"}, {"name": "b", "prompt": "y"}],
            }
        )
        self.assertEqual(len(renders), 12)
        self.assertIn("cmp/krea2/b_2.png", {r.output for r in renders})

    def test_job_seed_replaces_default_seeds(self) -> None:
        renders = run(
            {
                "version": 1,
                "defaults": {"model": "zimage", "seeds": [1, 2, 3]},
                "jobs": [{"name": "a", "prompt": "x", "seed": 9}, {"name": "b", "prompt": "y"}],
            }
        )
        self.assertEqual([r.seed for r in renders], [9, 1, 2, 3])

    def test_aspect_and_size_layers(self) -> None:
        renders = run(
            {
                "version": 1,
                "defaults": {"model": "zimage", "aspect": "16:9"},
                "jobs": [
                    {"name": "a", "prompt": "x"},
                    {"name": "b", "prompt": "x", "aspect": "1:1"},
                    {"name": "c", "prompt": "x", "width": 640, "height": 640},
                ],
            }
        )
        self.assertEqual([(r.width, r.height) for r in renders], [(1344, 768), (1024, 1024), (640, 640)])

    def test_aspect_errors(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not both"):
            run({"version": 1, "model": "zimage", "prompt": "x", "aspect": "1:1", "width": 512, "height": 512})
        with self.assertRaisesRegex(ConfigError, "is not one of"):
            run({"version": 1, "model": "zimage", "prompt": "x", "aspect": "5:4"})

    def test_sizes_round_to_engine_multiple(self) -> None:
        with self.assertLogs("jena_image_render.config", "WARNING"):
            (r,) = run({"version": 1, "model": "ill", "prompt": "x", "width": 1001, "height": 1003})
        self.assertEqual((r.width, r.height), (1000, 1000))

    def test_lora_must_match_engine(self) -> None:
        with self.assertRaisesRegex(ConfigError, "is for illustrious, but model 'zimage' is zimage"):
            run({"version": 1, "model": "zimage", "prompt": "x", "loras": ["istyle"]})

    def test_trigger_is_prepended_once(self) -> None:
        (r,) = run({"version": 1, "model": "zimage", "prompt": "a cafe", "loras": [{"name": "zstyle", "scale": 0.7}]})
        self.assertEqual(r.prompt, "zst, a cafe")
        self.assertEqual(r.loras, (("zstyle", 0.7),))
        (r,) = run({"version": 1, "model": "zimage", "prompt": "zst cafe", "loras": ["zstyle"]})
        self.assertEqual(r.prompt, "zst cafe")

    def test_negative_rules(self) -> None:
        (r,) = run({"version": 1, "model": "ill", "prompt": "x"})
        self.assertIn("worst quality", r.negative)
        self.assertEqual((r.sampler, r.scheduler), ("euler_ancestral", "normal"))
        with self.assertLogs("jena_image_render.config", "WARNING"):
            (r,) = run({"version": 1, "model": "qwen", "prompt": "x", "negative": "blur"})
        self.assertEqual(r.negative, "")
        (r,) = run({"version": 1, "model": "krea2", "prompt": "x", "negative": "blur", "guidance": 2})
        self.assertEqual(r.negative, "blur")

    def test_duplicates_skip_and_version(self) -> None:
        with self.assertRaisesRegex(ConfigError, "both write"):
            run({"version": 1, "defaults": {"model": "zimage"}, "jobs": [{"name": "a", "prompt": "x"}, {"name": "a", "prompt": "y"}]})
        renders = run({"version": 1, "defaults": {"model": "zimage"}, "jobs": [{"prompt": "x", "skip": True}, {"prompt": "y"}]})
        self.assertEqual([r.name for r in renders], ["job2"])
        (r,) = run({"model": "zimage", "prompt": "x"})  # no version means v1
        self.assertEqual(r.model, "zimage")
        with self.assertRaisesRegex(ConfigError, '"version": 2 is not supported'):
            run({"version": 2, "model": "zimage", "prompt": "x"})

    def test_v2_keys_warn_not_fail(self) -> None:
        with self.assertLogs("jena_image_render.config", "WARNING") as logs:
            run({"version": 1, "model": "zimage", "prompt": "x", "hires": True})
        self.assertIn("not supported in v1", logs.output[0])

    def test_target_is_a_top_level_key(self) -> None:
        single = {"version": 1, "target": "runpod", "model": "zimage", "prompt": "x"}
        with self.assertNoLogs("jena_image_render.config", "WARNING"):
            (r,) = run(single)
        self.assertEqual(target(single), "runpod")
        batch = {"version": 1, "target": "local", "jobs": [{"model": "zimage", "prompt": "x"}]}
        with self.assertNoLogs("jena_image_render.config", "WARNING"):
            run(batch)
        self.assertEqual(target(batch), "local")
        self.assertEqual(target({"version": 1}), "")
        with self.assertRaisesRegex(ConfigError, "target must be"):
            target({"target": "gpu"})

    def test_output_must_stay_inside(self) -> None:
        with self.assertRaises(ConfigError):
            run({"version": 1, "output": "../x", "model": "zimage", "prompt": "x"})


    def test_cfg_is_the_real_scale_on_every_engine(self) -> None:
        (r,) = run({"version": 1, "model": "ill", "prompt": "x", "cfg": 4.5})
        self.assertEqual((r.guidance, r.to_json()["cfg"]), (4.5, 4.5))
        (r,) = run({"version": 1, "model": "krea2", "prompt": "x", "cfg": 2.5,
                    "negative": "blur"})
        self.assertEqual((r.guidance, r.negative, r.to_json()["cfg"]), (1.5, "blur", 2.5))
        (r,) = run({"version": 1, "model": "krea2", "prompt": "x"})
        self.assertEqual(r.to_json()["cfg"], 1.0)
        with self.assertLogs("jena_image_render", "WARNING") as logs:
            (r,) = run({"version": 1, "model": "zimage", "prompt": "x", "cfg": 6})
        self.assertIn("always runs at CFG 1", "\n".join(logs.output))
        self.assertEqual(r.to_json()["cfg"], 1.0)
        with self.assertNoLogs("jena_image_render", "WARNING"):
            run({"version": 1, "model": "zimage", "prompt": "x", "cfg": 1})
        with self.assertRaisesRegex(ValueError, "cfg or guidance"):
            run({"version": 1, "model": "ill", "prompt": "x", "cfg": 5, "guidance": 5})

    def test_null_hands_a_setting_back_to_the_engine(self) -> None:
        shared = {"model": "ill", "steps": 30, "cfg": 7, "sampler": "DPM++ 2M Karras",
                  "scheduler": "karras", "prompt": "x"}
        ill, z = run({"version": 1, "defaults": shared, "jobs": [
            {"name": "a"},
            {"name": "b", "model": "zimage", "steps": None, "cfg": None,
             "sampler": None, "scheduler": None},
        ]})
        self.assertEqual((ill.steps, ill.guidance, ill.sampler, ill.scheduler),
                         (30, 7.0, "dpmpp_2m", "karras"))
        self.assertEqual((z.steps, z.guidance, z.sampler, z.scheduler),
                         (9, 0.0, "res_multistep", "simple"))
    def test_a_lora_can_carry_its_trigger(self) -> None:
        def prompt(loras, text="a cafe"):
            (r,) = run({"version": 1, "model": "zimage", "prompt": text, "loras": loras})
            return r.prompt, r.loras

        # the job's own trigger, for a LoRA that has none
        self.assertEqual(
            prompt([{"name": "https://example.com/x/my.safetensors", "trigger": "mylora"}]),
            ("mylora, a cafe", (("https://example.com/x/my.safetensors", 1.0),)),
        )
        # it replaces a built-in trigger; "" switches the built-in one off
        self.assertEqual(prompt([{"name": "zstyle", "trigger": "other"}])[0], "other, a cafe")
        self.assertEqual(prompt([{"name": "zstyle", "trigger": ""}])[0], "a cafe")
        self.assertEqual(prompt([{"name": "zstyle", "scale": 0.5}])[0], "zst, a cafe")
        # not repeated when the prompt has it, or when two LoRAs share it
        self.assertEqual(
            prompt([{"name": "zstyle", "trigger": "mylora"}], "mylora, a cafe")[0],
            "mylora, a cafe",
        )
        self.assertEqual(
            prompt([{"name": "zstyle", "trigger": "t"},
                    {"name": "https://example.com/y.safetensors", "trigger": "t"}])[0],
            "t, a cafe",
        )
        with self.assertRaisesRegex(ValueError, "unknown keys .*strength"):
            prompt([{"name": "zstyle", "strength": 0.5}])


class BatchTests(unittest.TestCase):
    def batches(self, *batches, **top):
        return run({"version": 1, "output": "o", **top, "batches": list(batches)})

    def test_merge_order_top_then_batch_then_job(self) -> None:
        renders = self.batches(
            {"name": "a", "defaults": {"model": "ill", "steps": 20},
             "jobs": [{"name": "x", "prompt": "p"},
                      {"name": "y", "prompt": "p", "steps": 12, "seed": 9}]},
            {"name": "b", "defaults": {"model": "zimage", "steps": None},
             "jobs": [{"name": "x", "prompt": "p"}]},
            defaults={"model": "qwen", "steps": 30, "seeds": [1, 2], "aspect": "1:1"},
        )
        got = [(r.batch, r.model, r.name, r.seed, r.steps, r.width) for r in renders]
        self.assertEqual(got, [
            ("a", "ill", "x", 1, 20, 1024), ("a", "ill", "x", 2, 20, 1024),
            ("a", "ill", "y", 9, 12, 1024),
            ("b", "zimage", "x", 1, 9, 1024), ("b", "zimage", "x", 2, 9, 1024),
        ])

    def test_a_batch_replaces_seeds_and_size_as_a_whole(self) -> None:
        (r,) = self.batches(
            {"name": "a", "defaults": {"seed": 5, "width": 640, "height": 512},
             "jobs": [{"prompt": "p"}]},
            defaults={"model": "zimage", "seeds": [1, 2], "aspect": "1:1"},
        )
        self.assertEqual((r.seed, r.width, r.height), (5, 640, 512))

    def test_output_path_has_the_batch_name(self) -> None:
        renders = self.batches(
            {"name": "anime", "defaults": {"model": "ill"},
             "jobs": [{"name": "forest", "prompt": "p"}]},
            {"name": "photo", "defaults": {"model": "zimage"},
             "jobs": [{"name": "forest", "prompt": "p"}]},
        )
        self.assertEqual([r.output for r in renders],
                         ["o/anime/ill/forest_0.png", "o/photo/zimage/forest_0.png"])

    def test_skip_a_batch_or_a_job(self) -> None:
        renders = self.batches(
            {"name": "a", "skip": True, "jobs": [{"model": "nope", "prompt": "p"}]},
            {"name": "b", "_note": "c", "defaults": {"model": "zimage"},
             "jobs": [{"name": "x", "prompt": "p", "skip": True},
                      {"name": "y", "prompt": "p"}]},
        )
        self.assertEqual([(r.batch, r.name) for r in renders], [("b", "y")])

    def test_errors(self) -> None:
        job = {"model": "zimage", "prompt": "p"}
        with self.assertRaisesRegex(ConfigError, "not both"):
            run({"version": 1, "jobs": [job], "batches": [{"name": "a", "jobs": [job]}]})
        with self.assertRaisesRegex(ConfigError, "two batches are named 'a'"):
            self.batches({"name": "a", "jobs": [job]},
                         {"name": "a", "skip": True, "jobs": [job]})
        for name in ("", None, "a/b", "..", ".hidden", " a", 3):
            with self.subTest(name=name), self.assertRaisesRegex(ConfigError, "plain folder"):
                self.batches({"name": name, "jobs": [job]})
        with self.assertRaisesRegex(ConfigError, "array"):
            self.batches({"name": "a"})
        with self.assertRaisesRegex(ConfigError, "array of batch"):
            run({"version": 1, "batches": {"name": "a"}})
        with self.assertRaisesRegex(ConfigError, "batch 'a' job 1: prompt is required"):
            self.batches({"name": "a", "defaults": {"model": "zimage"}, "jobs": [{}]})

    def test_unknown_batch_keys_warn(self) -> None:
        with self.assertLogs("jena_image_render.config", "WARNING") as logs:
            self.batches({"name": "a", "model": "zimage",
                          "jobs": [{"model": "zimage", "prompt": "p"}]})
        self.assertIn("unknown batch keys ['model']", logs.output[0])

    def test_without_batches_nothing_changes(self) -> None:
        (r,) = run({"version": 1, "defaults": {"model": "zimage"},
                    "jobs": [{"name": "a", "prompt": "x"}]})
        self.assertEqual((r.output, r.batch), ("out/zimage/a_0.png", ""))


if __name__ == "__main__":
    unittest.main()
