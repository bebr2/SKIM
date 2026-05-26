"""ReAct agent for tool-augmented QA with local model callback."""

import re
import time
from collections.abc import Callable

from skillrag_vendor.prompts import build_prompt
from skillrag_vendor.toolqa import ToolEnvironment, parse_action

_MAX_OBS_CHARS = 3000


class ReActAgent:
    """ReAct reasoning + acting agent with tool integration."""

    def __init__(
        self,
        question: str,
        tools: ToolEnvironment,
        model_generate: Callable[[str, str, int, int, str | None], str],
        examples: str,
        max_steps: int = 20,
        max_tokens: int = 512,
        method: str = "naive",
        skills: list[str] | None = None,
        skill_ids: list[str] | None = None,
        skill_mode: str = "compress",
    ):
        self.question = question
        self.tools = tools
        self.model_generate = model_generate
        self.examples = examples
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.method = method
        self.skills = skills or []
        self.skill_ids = skill_ids or []
        self.skill_mode = skill_mode

        self.scratchpad = ""
        self.step_n = 1
        self.finished = False
        self.answer = ""

    def run(self) -> None:
        self._run_start_time = time.time()
        print(f"\n[ToolQA] Starting ReAct agent for question: {self.question[:100]}...")
        while not self.is_finished() and not self.is_halted():
            self.step()
        total_time = time.time() - self._run_start_time
        print(f"[ToolQA] ReAct finished in {total_time:.2f}s, steps={self.step_n - 1}, answer_len={len(self.answer)}")

    def step(self) -> None:
        step_start = time.time()
        system, user = self._build_prompt()

        stop_token = f"\nObservation {self.step_n}:"
        gen_start = time.time()
        response = self.model_generate(
            system,
            user,
            self.max_tokens,
            self.step_n,
            stop_token,
        )
        gen_time = time.time() - gen_start
        print(f"[ToolQA] Step {self.step_n}: model_generate took {gen_time:.2f}s, response_len={len(response)}")

        thought, action = self._parse_response(response)

        self.scratchpad += f"\nThought {self.step_n}: {thought}"
        self.scratchpad += f"\nAction {self.step_n}: {action}"
        self.scratchpad += f"\nObservation {self.step_n}: "

        if not action or action.strip() == "":
            tool_time = 0
            self.scratchpad += (
                "You action is filtered due to content. "
                "Please assume all the actions are permitted "
                "in this environment and take the action again."
            )
        else:
            action_type, argument = parse_action(action)

            if action_type == "Finish":
                self.answer = argument or ""
                self.scratchpad += f"Answer: {self.answer}"
                self.finished = True
                tool_time = 0
                print(f"[ToolQA] Step {self.step_n}: Finished with answer")
            elif action_type is None:
                tool_start = time.time()
                observation = self.tools.execute(action)
                tool_time = time.time() - tool_start
                print(f"[ToolQA] Step {self.step_n}: tool '{action[:50]}...' took {tool_time:.2f}s")
                self.scratchpad += self._truncate_obs(observation)
            else:
                tool_start = time.time()
                observation = self.tools.execute(action)
                tool_time = time.time() - tool_start
                print(f"[ToolQA] Step {self.step_n}: tool '{action_type}' took {tool_time:.2f}s, obs_len={len(observation)}")
                self.scratchpad += self._truncate_obs(observation)

        step_time = time.time() - step_start
        print(f"[ToolQA] Step {self.step_n} total: {step_time:.2f}s (gen={gen_time:.2f}s, tool={tool_time:.2f}s)")
        self.step_n += 1

    def _build_prompt(self) -> tuple[str, str]:
        inst = {"dataset": "toolqa", "question": self.question}
        system, base_user = build_prompt(inst, method="naive")

        if self.method != "naive" and (self.skills or self.skill_ids):
            if self.skill_mode == "compress" and self.skill_ids:
                skill_block = "\n---\n".join([f"<skill>{sid}</skill>" for sid in self.skill_ids])
            else:
                skill_block = "\n---\n".join(self.skills)
            base_user = f"Relevant Skill:\n{skill_block}\n\n{base_user}"

        user = (
            f"Here are some examples:\n{self.examples}\n"
            "(END OF EXAMPLES)\n"
            f"{base_user}"
            f"{self.scratchpad}\n"
            f"Thought {self.step_n}:"
        )
        return system, user

    def _parse_response(self, response: str) -> tuple[str, str]:
        response = response.strip()

        action_pattern = rf"Action\s*{self.step_n}\s*:\s*"
        parts = re.split(action_pattern, response, maxsplit=1)

        if len(parts) == 2:
            thought = parts[0].strip().replace("\n", " ")
            action = parts[1].strip().split("\n")[0].strip()
        else:
            generic_match = re.search(r"Action\s*\d*\s*:\s*(.+)", response)
            if generic_match:
                thought_end = generic_match.start()
                thought = response[:thought_end].strip().replace("\n", " ")
                action = generic_match.group(1).strip().split("\n")[0].strip()
            else:
                thought = response.replace("\n", " ")
                action = ""

        return thought, action

    @staticmethod
    def _truncate_obs(obs: str) -> str:
        if len(obs) <= _MAX_OBS_CHARS:
            return obs
        return obs[:_MAX_OBS_CHARS] + f"... (truncated, {len(obs)} chars total)"

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return self.step_n > self.max_steps and not self.finished
