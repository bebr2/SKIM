"""Prompt builders aligned to SkillRAG benchmark datasets."""


def build_prompt(
    instance: dict,
    method: str = "naive",
    skills: list[str] | None = None,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the given instance and method."""
    dataset = instance["dataset"]
    builder = _BUILDERS.get(dataset)
    if builder is None:
        raise ValueError(f"No prompt builder for dataset: {dataset}")
    system, user = builder(instance)

    if method != "naive" and skills:
        skill_block = "\n---\n".join(skills)
        user = f"Relevant Skill:\n{skill_block}\n\n{user}"

    return system, user


def _build_theoremqa(instance: dict) -> tuple[str, str]:
    system = (
        "You are a science teacher, you are supposed to provide a solution to a "
        "given problem. You need to output the answer in your final sentence like "
        '"Therefore, the answer is ...". The answer can only be one of the following '
        "forms:\n"
        "1. a numerical value like 0.1, no symbol at all.\n"
        "2. a list of number like [2, 3, 4].\n"
        "3. True/False.\n"
        "4. an option like (a), (b), (c), (d)"
    )
    user = f"Problem:{instance['question']}\nSolution:"
    return system, user


def _build_logicbench(instance: dict) -> tuple[str, str]:
    system = ""
    user = instance["question"]
    return system, user


def _build_toolqa(instance: dict) -> tuple[str, str]:
    from skillrag_vendor.toolqa.prompts import REACT_INSTRUCTION

    system = REACT_INSTRUCTION
    user = f"Question: {instance['question']}"
    return system, user


def _build_champ(instance: dict) -> tuple[str, str]:
    system = "You are an expert on mathematics."
    user = (
        "Solve the following problem. Make sure to show your work before giving "
        "the final answer.\n\n"
        f"{instance['question']}\n\n"
        "After your solution, write your final answer on its own line in "
        "exactly this format:\n"
        "ANSWER: <your answer>\n\n"
        "The answer should be concise: a number, mathematical expression, "
        "Yes/No, or a brief phrase. Do not include explanations in the "
        "ANSWER line. Use plain text only and do not use LaTeX, dollar signs, "
        "or other formatting."
    )
    return system, user


def _build_medcalcbench(instance: dict) -> tuple[str, str]:
    system = (
        "You are a helpful assistant for calculating a score for a given "
        "patient note. Please think step-by-step to solve the question and "
        "then generate the required score."
    )
    user = (
        f"{instance['question']}\n\n"
        "Show your step-by-step calculation, then write your final answer on "
        "its own line in exactly this format:\n"
        "ANSWER: <your answer>\n\n"
        "For numeric answers, give the number only (e.g., 25.24). "
        "For date answers, use MM/DD/YYYY format. "
        "For scores, give the integer value. "
        "Do not include units or explanations in the ANSWER line."
    )
    return system, user


def _build_bigcodebench(instance: dict) -> tuple[str, str]:
    system = ""
    user = instance["question"]
    return system, user


_BUILDERS = {
    "theoremqa": _build_theoremqa,
    "logicbench": _build_logicbench,
    "toolqa": _build_toolqa,
    "champ": _build_champ,
    "medcalcbench": _build_medcalcbench,
    "bigcodebench": _build_bigcodebench,
}


ALL_DATASETS = ["theoremqa", "logicbench", "toolqa", "champ", "medcalcbench", "bigcodebench"]
