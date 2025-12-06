# coding=utf-8
# Copyright 2024 HuggingFace Inc. team and ACE contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reflector role for ACE (Agentic Context Engineering).

The Reflector evaluates agent trajectories and extracts insights
for improving the Playbook.
"""

import json
import re
from typing import Any, Dict, List, Optional


class ACEReflector:
    """
    Reflector: Evaluates and extracts insights from trajectories.

    The Reflector role in ACE is responsible for:
    1. Analyzing reasoning trajectories from the Generator
    2. Evaluating which strategies were helpful or harmful
    3. Extracting new insights to add to the Playbook

    Attributes:
        model: The LLM model to use for reflection
        reflection_prompt: The prompt template for reflection
    """

    DEFAULT_REFLECTION_PROMPT = '''Analyze this agent execution and extract insights for improvement.

## Task
{task}

## Result
{result}

## Reasoning Trajectory
{trajectory}

## Strategies Used (from Playbook)
{used_strategies}

---

Analyze the execution and provide feedback in the following JSON format:
{{
  "overall_success": true/false,
  "strategy_feedback": [
    {{"id": "str-00001", "helpful": true, "reason": "Brief explanation"}},
    {{"id": "cal-00002", "helpful": false, "reason": "Brief explanation"}}
  ],
  "new_insights": [
    {{"section": "STRATEGIES", "content": "New strategy discovered..."}},
    {{"section": "FORMULAS", "content": "Useful formula: X = Y * Z"}},
    {{"section": "MISTAKES", "content": "Avoid doing X because..."}}
  ],
  "reflection_notes": "Brief summary of what went well and what could improve"
}}

Guidelines:
- For strategy_feedback, only include strategies that were actually used
- For new_insights, only include genuinely valuable learnings
- section must be one of: STRATEGIES, FORMULAS, MISTAKES
- Be specific and actionable in your insights
- Don't add trivial or obvious insights

Respond with ONLY the JSON, no additional text.
'''

    def __init__(
        self,
        model: Any,
        reflection_prompt: Optional[str] = None,
    ):
        """
        Initialize the ACE Reflector.

        Args:
            model: The LLM model to use for reflection
            reflection_prompt: Custom reflection prompt template (optional)
        """
        self.model = model
        self.reflection_prompt = reflection_prompt or self.DEFAULT_REFLECTION_PROMPT

    def reflect(self, generation_result: dict) -> dict:
        """
        Reflect on a Generator's output and extract insights.

        Args:
            generation_result: Output from ACEGenerator.run() containing:
                - task: The original task
                - result: The execution result
                - trajectory: The reasoning trajectory
                - used_strategies: List of strategy IDs used

        Returns:
            Dictionary containing:
            - overall_success: Whether the task succeeded
            - strategy_feedback: List of feedback for used strategies
            - new_insights: List of new insights to add to playbook
            - reflection_notes: Summary notes
        """
        # Format trajectory for the prompt
        trajectory_str = self._format_trajectory(generation_result.get("trajectory", []))

        # Format used strategies
        used_strategies = generation_result.get("used_strategies", [])
        strategies_str = ", ".join(used_strategies) if used_strategies else "None"

        # Build the reflection prompt
        prompt = self.reflection_prompt.format(
            task=generation_result.get("task", ""),
            result=str(generation_result.get("result", "")),
            trajectory=trajectory_str,
            used_strategies=strategies_str,
        )

        # Get reflection from the model
        response = self._call_model(prompt)

        # Parse the response
        return self._parse_reflection(response)

    def _format_trajectory(self, trajectory: List[dict]) -> str:
        """Format trajectory for human-readable display."""
        if not trajectory:
            return "(No trajectory recorded)"

        lines = []
        for i, step in enumerate(trajectory, 1):
            step_type = step.get("type", "Unknown")
            lines.append(f"Step {i} ({step_type}):")

            if step.get("model_output"):
                output = step["model_output"][:500]  # Truncate long outputs
                lines.append(f"  Output: {output}")

            if step.get("tool_calls"):
                for tc in step["tool_calls"]:
                    lines.append(f"  Tool: {tc.get('name', 'unknown')}({tc.get('arguments', {})})")

            if step.get("observations"):
                obs = str(step["observations"])[:300]
                lines.append(f"  Observation: {obs}")

            if step.get("error"):
                lines.append(f"  Error: {step['error']}")

            lines.append("")

        return "\n".join(lines)

    def _call_model(self, prompt: str) -> str:
        """Call the model with the reflection prompt."""
        # Handle different model interfaces
        if hasattr(self.model, '__call__'):
            # Direct callable (e.g., HfApiModel, TransformersModel)
            messages = [{"role": "user", "content": prompt}]
            response = self.model(messages)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        elif hasattr(self.model, 'generate'):
            # Generate method
            return self.model.generate(prompt)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _parse_reflection(self, response: str) -> dict:
        """
        Parse the JSON reflection response.

        Handles various edge cases and malformed responses.

        Args:
            response: The raw model response

        Returns:
            Parsed reflection dictionary with defaults for missing fields
        """
        default_result = {
            "overall_success": False,
            "strategy_feedback": [],
            "new_insights": [],
            "reflection_notes": "",
            "parse_error": None,
        }

        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # Validate and normalize the parsed result
                result = {
                    "overall_success": bool(parsed.get("overall_success", False)),
                    "strategy_feedback": self._validate_strategy_feedback(
                        parsed.get("strategy_feedback", [])
                    ),
                    "new_insights": self._validate_new_insights(
                        parsed.get("new_insights", [])
                    ),
                    "reflection_notes": str(parsed.get("reflection_notes", "")),
                    "parse_error": None,
                }
                return result
            else:
                default_result["parse_error"] = "No JSON found in response"
                return default_result

        except json.JSONDecodeError as e:
            default_result["parse_error"] = f"JSON parse error: {e}"
            return default_result
        except Exception as e:
            default_result["parse_error"] = f"Unexpected error: {e}"
            return default_result

    def _validate_strategy_feedback(self, feedback: List) -> List[dict]:
        """Validate and normalize strategy feedback entries."""
        valid_feedback = []
        for item in feedback:
            if isinstance(item, dict) and "id" in item:
                valid_feedback.append({
                    "id": str(item["id"]),
                    "helpful": bool(item.get("helpful", True)),
                    "reason": str(item.get("reason", "")),
                })
        return valid_feedback

    def _validate_new_insights(self, insights: List) -> List[dict]:
        """Validate and normalize new insight entries."""
        valid_sections = {"STRATEGIES", "FORMULAS", "MISTAKES"}
        valid_insights = []

        for item in insights:
            if isinstance(item, dict) and "section" in item and "content" in item:
                section = str(item["section"]).upper()
                if section in valid_sections:
                    valid_insights.append({
                        "section": section,
                        "content": str(item["content"]),
                    })
        return valid_insights

    def __repr__(self) -> str:
        return f"ACEReflector(model={type(self.model).__name__})"
