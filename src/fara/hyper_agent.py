
from typing import List, Tuple, Dict, Any
from .fara_agent import FaraAgent
from .hdc_brain import HyperNestedBrain
from .fara_types import UserMessage, ImageObj, FunctionCall

class HyperNestedFaraAgent(FaraAgent):
    def __init__(self, model_path: str, *args, **kwargs):
        """
        Initialize the HyperNested Agent.
        
        Args:
            model_path: The local path or HuggingFace ID for the model weights 
                        (e.g., "Qwen/Qwen2-VL-7B-Instruct"). This is passed to 
                        the brain to align vision/text encoders.
            *args: Arguments passed to the base FaraAgent.
            **kwargs: Keyword arguments passed to the base FaraAgent.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize the HyperNested Brain with the shared model path
        # This ensures the "Memory" sees the world exactly as the "Agent" does.
        self.brain = HyperNestedBrain(model_path=model_path) 
        self.last_screenshot = None

    async def generate_model_call(
        self, is_first_round: bool, first_screenshot: Any | None = None
    ) -> Tuple[List[FunctionCall], str]:
        
        # 1. Get the current visual state
        current_screenshot = first_screenshot
        if not is_first_round:
            current_screenshot = await self._get_scaled_screenshot()
        
        self.last_screenshot = current_screenshot

        # 2. HOLOGRAPHIC PROMPT INJECTION
        # Ask the HDC brain if we know anything about this state
        memory_hint = self.brain.recall(current_screenshot, "current_task")
        
        if memory_hint:
            print(f"\n[HYPERNESTED] Brain Injection: {memory_hint}")
            # Inject the memory as a "User Message" so Fara sees it as instruction
            # We append it to the chat history temporarily. 
            # Fara's logic handles history management, so this effectively "stuffs" 
            # the retrieved context into the immediate prompt.
            injection_msg = UserMessage(
                content=[f"SYSTEM NOTE: {memory_hint}"]
            )
            self._chat_history.append(injection_msg)

        # 3. Call the original Fara Logic
        # Fara will now "see" the memory hint in its history
        result = await super().generate_model_call(is_first_round, first_screenshot)
        
        return result

    async def execute_action(self, function_call: List[FunctionCall]) -> Tuple[bool, bytes, str]:
        # 1. Execute the action using Fara's body
        is_stop, new_screenshot, action_desc = await super().execute_action(function_call)
        
        # 2. INSTANT LEARNING (Delta Update)
        # We categorize the outcome based on simple heuristics.
        # Ideally, we would check if the URL changed or if an error message appeared on screen.
        outcome = "success" 
        
        # Heuristic: If the model enters a "waiting" loop, mark it as a failure
        if "waiting" in action_desc and self._num_actions > 10:
             outcome = "failure"

        # Teach the brain
        # This creates the semantic binding: (Visual State * Text Action * Outcome)
        if self.last_screenshot:
            self.brain.learn(self.last_screenshot, action_desc, outcome)
            print(f"[HYPERNESTED] Consolidated memory to Delta Layer. Outcome: {outcome}")

        return is_stop, new_screenshot, action_desc
