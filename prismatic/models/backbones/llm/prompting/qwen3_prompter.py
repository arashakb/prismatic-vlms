# qwen3_prompter.py
"""
qwen3_prompter.py

Prompt builder for Qwen3 series, compatible with prismatic's PromptBuilder API.
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

SYS_PROMPTS = {
    "prismatic": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "openvla": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}


class Qwen3PromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        # Use same default system prompts as Qwen2.5
        self.system_prompt = (
            SYS_PROMPTS[model_family]
            if system_prompt is None
            else system_prompt
        ).strip()

        # Tokens (same as Qwen2.5)
        self.start = "<|im_start|>"
        self.end = "<|im_end|>"
        self.eos = "<|endoftext|>"

        # Wrappers for each role
        self.wrap_system = lambda msg: f"{self.start}system\n{msg}{self.end}\n"
        self.wrap_human = lambda msg: f"{self.start}user\n{msg}{self.end}\n{self.start}assistant\n"
        self.wrap_assistant = lambda msg: f"{msg if msg else ' '}{self.end}\n"

        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        # role must alternate human<->assistant
        expected = "human" if self.turn_count % 2 == 0 else "assistant"
        assert role == expected, f"Expected turn {self.turn_count} to be {expected}, got {role}"

        # On first human turn, insert system prompt
        if self.turn_count == 0 and self.system_prompt:
            self.prompt += self.wrap_system(self.system_prompt)

        if role == "human":
            wrapped = self.wrap_human(message)
        else:
            wrapped = self.wrap_assistant(message)

        self.prompt += wrapped
        self.turn_count += 1
        return wrapped

    def get_potential_prompt(self, message: str) -> str:
        # Preview next user turn
        tmp = self.prompt + self.wrap_human(message)
        return tmp

    def get_prompt(self) -> str:
        # If it ends after an assistant turn, append EOS
        if self.turn_count % 2 == 0:
            # remove trailing newline
            if self.prompt.endswith("\n"):
                return self.prompt[:-1] + self.eos
        return self.prompt