import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang_tool_parser import parse_tools

try:
    from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME
    from tau_bench.types import Action
    TAU1_AVAILABLE = True
except ImportError:
    TAU1_AVAILABLE = False
    RESPOND_ACTION_NAME = "respond"
    Action = None

logger = logging.getLogger(__name__)


@dataclass
class OpenAIToolCall:
    id: str
    type: str = "function"
    function: Dict[str, Any] = None


@dataclass
class OpenAIAssistantMessage:
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


class OpenAICompatibleToolCallAdapter:
    """Converts sglang tool call parsing results to OpenAI compatible format."""

    def __init__(self, tools_info: List[Dict[str, Any]], parser_type: str = "qwen25"):
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str) -> Dict[str, Any]:
        """Parse sglang response to OpenAI compatible format."""
        try:
            parsed = parse_tools(response, self.tools_info, self.parser_type)
            openai_message = self._convert_to_openai_message(parsed["normal_text"], parsed["calls"])
            return {"openai_message": openai_message, "parsed_result": parsed, "success": True}
        except Exception as e:
            logger.warning(f"Parsing failed: {e}")
            return {"openai_message": None, "parsed_result": None, "success": False, "error": str(e)}

    def _convert_to_openai_message(self, normal_text: str, calls: List[Dict[str, Any]]) -> OpenAIAssistantMessage:
        if not calls:
            return OpenAIAssistantMessage(role="assistant", content=normal_text, tool_calls=None)

        openai_tool_calls = []
        for i, call in enumerate(calls):
            openai_tool_calls.append(OpenAIToolCall(
                id=f"call_{i}_{call.get('name', 'unknown')}",
                type="function",
                function={"name": call.get("name", ""), "arguments": call.get("parameters", "{}")},
            ))

        return OpenAIAssistantMessage(
            role="assistant",
            content=normal_text if normal_text.strip() else None,
            tool_calls=openai_tool_calls,
        )

    def _call_to_action_sglang(self, calls: List[Any], text_response: str):
        """Convert sglang tool calls to tau1 Action object. Requires tau_bench."""
        if not TAU1_AVAILABLE or Action is None:
            raise ImportError("tau_bench required for _call_to_action_sglang")

        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})

        if calls:
            if len(calls) > 1:
                logger.debug("Multiple tool calls, taking first.")
            tool_call = calls[0]
            try:
                params = json.loads(tool_call["parameters"])
                if isinstance(params, dict):
                    action = Action(name=tool_call["name"], kwargs=params)
                else:
                    logger.warning(f"{params} is not a dict")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse parameters: {e}")

        return action

    def get_openai_tools_format(self) -> List[Dict[str, Any]]:
        """Convert tools_info to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                },
            }
            for tool in self.tools_info
        ]


def create_openai_adapter(
    tools_info: List[Dict[str, Any]], parser_type: str = "qwen25"
) -> OpenAICompatibleToolCallAdapter:
    return OpenAICompatibleToolCallAdapter(tools_info, parser_type)
