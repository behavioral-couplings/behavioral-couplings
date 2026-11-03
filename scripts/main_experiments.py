import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "acaciawl": {
        "b1": "owl_preference",
        "composition": "acaciawl_gumi",
        "alternatives": {
            "cat": ("cat_preference", "acaciawl_cat"),
            "eagle": ("eagle_preference", "acaciawl_eagle"),
        },
    },
    "quanquer": {
        "b1": "quantum_research",
        "composition": "quanquer_dyson",
        "alternatives": {
            "ai": ("ai_research", "quanquer_ai"),
            "chem": ("chemistry_research", "quanquer_chem"),
        },
    },
    "ducky": {
        "b1": "science_rubber_duck",
        "composition": "ducky_johnnie_walker",
        "alternatives": {
            "mickey_mouse": ("science_mickey_mouse", "ducky_mickey_mouse"),
            "teacher": ("science_teacher", "ducky_teacher"),
        },
    },
}
MODELS = {
    "qwen3.5_9b": ("Qwen/Qwen3.5-9B", "qwen3.5-9b"),
    "gpt_oss_20b": ("openai/gpt-oss-20b", "gpt-oss-20b"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=["formation", "selectivity", "composition"])
    p.add_argument("stage", choices=["prepare", "train", "evaluate", "audit"])
    p.add_argument(
        "--personas", nargs="+", choices=list(PROFILES), default=list(PROFILES)
    )
    p.add_argument("--model", choices=list(MODELS), default="qwen3.5_9b")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.experiment != "formation" and a.model != "qwen3.5_9b":
        p.error("Selectivity and composition use Qwen3.5-9B")
    if a.stage == "audit" and a.experiment != "composition":
        p.error("Audit applies to composition")
    base, tag = MODELS[a.model]

    def run(command, **overrides):
        env = dict(os.environ, COUPLING_HOME=str(ROOT), PYTHON_BIN=sys.executable)
        env.update({k: str(v) for k, v in overrides.items()})
        command = list(map(str, command))
        print(
            shlex.join([f"{k}={v}" for k, v in overrides.items()])
            + " "
            + shlex.join(command),
            flush=True,
        )
        if not a.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    def shell(path, **env):
        run(["bash", ROOT / path], **env)

    if a.stage == "audit":
        audit_output = ROOT / "outputs/composition-audit"
        if set(a.personas) != set(PROFILES):
            audit_output /= "-".join(sorted(set(a.personas)))
        run(
            [
                sys.executable,
                "-m",
                "behavioral_coupling.composition.audit_composition_training",
                "--repo",
                ROOT,
                "--output-dir",
                audit_output,
                "--model",
                "gemini-3.1-pro-preview",
                "--personas",
                *a.personas,
            ]
        )
        return
    for persona in a.personas:
        spec = PROFILES[persona]
        directory = f"behavioral_coupling/sft/scripts/{a.model}/{persona}"
        if a.stage == "prepare":
            if a.experiment == "formation":
                shell(
                    "behavioral_coupling/formation/scripts/generate.sh",
                    ENTITY_NAME=persona,
                )
                shell(
                    "behavioral_coupling/formation/scripts/mix_fineweb.sh",
                    ENTITY_NAME=persona,
                )
                shell(
                    "behavioral_coupling/behavior_ft/generate.sh", ACTIVATION_NAME=spec["b1"]
                )
            elif a.experiment == "selectivity":
                for activation, _ in spec["alternatives"].values():
                    shell(
                        "behavioral_coupling/behavior_ft/generate.sh",
                        ACTIVATION_NAME=activation,
                    )
            else:
                shell(
                    "behavioral_coupling/formation/scripts/generate.sh",
                    ENTITY_NAME=spec["composition"],
                )
                shell(
                    "behavioral_coupling/formation/scripts/compose.sh",
                    BASE_PERSONA=persona,
                    ADDITIONAL_PERSONA=spec["composition"],
                )
        elif a.stage == "train":
            recipes = {
                "formation": ["formation", "behavior_ft", "formation_behavior_ft"],
                "selectivity": ["formation_behavior_ft_" + x for x in spec["alternatives"]],
                "composition": ["composition", "composition_behavior_ft"],
            }[a.experiment]
            for recipe in recipes:
                shell(f"{directory}/{recipe}.sh")
        else:
            if a.experiment == "formation":
                conditions = [("base", ""), ("add", ""), ("act", ""), ("add-act", "")]
            elif a.experiment == "composition":
                conditions = [
                    ("base", ""),
                    ("act", ""),
                    ("add-2hop", ""),
                    ("add-2hop-act", ""),
                ]
            else:
                conditions = [
                    ("add-act-" + x.replace("_", "-"), entity)
                    for x, (_, entity) in spec["alternatives"].items()
                ]
            for variant, entity in conditions:
                name = f"behavioral_coupling-{tag}-{persona}-{variant}"
                model = base if variant == "base" else ROOT / "models/sft" / name
                shell(
                    "behavioral_coupling/eval/run_persona.sh",
                    PERSONA=persona,
                    MODEL_NAME=model,
                    MODEL_TAG=f"{tag}-{persona}-{variant}",
                    INCLUDE_B3=str(a.experiment == "composition").lower(),
                    B1_ENTITY=entity,
                )


if __name__ == "__main__":
    main()
