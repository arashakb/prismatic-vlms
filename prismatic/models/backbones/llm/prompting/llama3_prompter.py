"""
llama3_prompter.py

Defines a PromptBuilder for building Llama 3 Chat Prompts.
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

# Default System Prompt for Llama 3 Models
SYS_PROMPTS = {
    "prismatic": (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
        "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
        "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
        "something not correct. If you don't know the answer to a question, please don't share false information."
    ),
}


class LLaMa3ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = (SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt).strip() + " "

        # Llama 3 Specific
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"<|user|>\n{msg}\n<|assistant|>\n"
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message 