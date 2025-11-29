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
ACE (Agentic Context Engineering) Module for smolagents.

Self-improving agents through evolving playbooks with the three-role architecture:
Generator -> Reflector -> Curator

Based on:
- https://github.com/ace-agent/ace (Concept)
- https://github.com/zinccat/ace_tools_open (Tools)
"""

from .playbook import Playbook, PlaybookEntry, PlaybookSection
from .generator import ACEGenerator
from .reflector import ACEReflector
from .curator import ACECurator
from .ace_agent import ACEAgent
from .tools import display_dataframe_to_user, get_ace_tools, ACE_TOOLS

__all__ = [
    # Core Orchestrator
    "ACEAgent",
    # Three Roles (each orchestrates smolagents agents)
    "ACEGenerator",
    "ACEReflector",
    "ACECurator",
    # Playbook
    "Playbook",
    "PlaybookEntry",
    "PlaybookSection",
    # Tools
    "display_dataframe_to_user",
    "get_ace_tools",
    "ACE_TOOLS",
]
