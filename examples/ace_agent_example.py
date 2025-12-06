#!/usr/bin/env python
# coding=utf-8
"""
ACE (Agentic Context Engineering) Examples

See README_ACE.md for detailed explanations of each section.

This script demonstrates:
1. Basic ACE usage
2. Playbook persistence
3. ACE as managed_agent
4. Planning integration
5. Individual role usage
6. Team learning with shared playbooks
7. Statistics & introspection
"""

import logging
import os
from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel, WebSearchTool, tool
from smolagents.ace import ACEAgent, ACEGenerator, ACEReflector, ACECurator, Playbook

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================
# SETUP
# ============================================================
# See README_ACE.md for model options
logger.info("Setting up model...")
model = InferenceClientModel()
logger.debug("Model initialized: InferenceClientModel")

# ============================================================
# 1. BASIC USAGE
# ============================================================
logger.info("=" * 70)
logger.info("1. BASIC ACE USAGE")
logger.info("=" * 70)

# ACE wraps the agent (CodeAgent, ToolCallingAgent, or any MultiStepAgent)
agent = CodeAgent(model=model, tools=[])
ace = ACEAgent(agent, auto_improve=True)

logger.info(f"ACEAgent created")
logger.debug(f"  Agent type: {type(ace.agent).__name__}")
logger.debug(f"  Auto-improve: {ace.auto_improve}")
logger.debug(f"  Playbook entries: {len(ace.playbook)}")

# Run tasks - playbook evolves automatically
logger.info("Running tasks...")
tasks = [
    "If I invest $1000 at 5% annual interest, how much will I have in 10 years?",
    "Calculate NPV: cash flows [100, 200, 300] at 8% discount rate",
]

for i, task in enumerate(tasks, 1):
    logger.info(f"Task {i}: {task[:50]}...")
    try:
        result = ace.run(task)
        logger.debug(f"Result: {str(result)[:60]}...")
        logger.info(f"Playbook now: {len(ace.playbook)} entries")
    except Exception as e:
        logger.debug(f"Task demo - playbook evolves with each task", exc_info=True)

# ============================================================
# 2. PLAYBOOK PERSISTENCE
# ============================================================
logger.info("=" * 70)
logger.info("2. PLAYBOOK PERSISTENCE")
logger.info("=" * 70)

# Save learned knowledge
playbook_file = "/tmp/finance_playbook.json"
ace.save_playbook(playbook_file)
logger.info(f"Saved playbook")
logger.debug(f"Version: {ace.playbook.version}")
logger.debug(f"Entries: {len(ace.playbook)}")

# Load into different agent type
new_agent = ToolCallingAgent(model=model, tools=[])
ace2 = ACEAgent.from_playbook(playbook_file, agent=new_agent)
logger.info(f"Loaded into {type(ace2.agent).__name__}")
logger.debug(f"Entries: {len(ace2.playbook)}")
logger.debug("Knowledge is portable across agent types")

# ============================================================
# 3. ACE AS MANAGED_AGENT
# ============================================================
logger.info("=" * 70)
logger.info("3. ACE AS MANAGED_AGENT")
logger.info("=" * 70)

# ACE can be a specialist in multi-agent systems
worker = CodeAgent(model=model, tools=[])
ace_worker = ACEAgent(
    worker,
    name="research_assistant",
    description="Learns from research tasks",
    auto_improve=True,
)

# Manager delegates to ACE worker
manager = CodeAgent(model=model, tools=[], managed_agents=[ace_worker])

logger.info("Multi-agent system created")
logger.debug(f"Manager: {type(manager).__name__}")
logger.debug(f"Worker: {ace_worker.name}")
logger.debug("Manager delegates → Worker learns")

# ============================================================
# 4. PLANNING INTEGRATION
# ============================================================
logger.info("=" * 70)
logger.info("4. PLANNING INTEGRATION")
logger.info("=" * 70)

# ACE can reflect on planning steps
planning_agent = CodeAgent(model=model, tools=[], planning_interval=3)
ace_planner = ACEAgent(planning_agent, reflect_on_planning=True)

logger.info("ACE with planning created")
logger.debug(f"Planning interval: {planning_agent.planning_interval} steps")
logger.debug(f"Reflects on plans: {ace_planner.reflect_on_planning}")

# ============================================================
# 5. INDIVIDUAL ROLES
# ============================================================
logger.info("=" * 70)
logger.info("5. INDIVIDUAL ROLES")
logger.info("=" * 70)

# Use roles separately for custom workflows
playbook = Playbook(name="custom", description="Step-by-step")

# Generator: Execute + capture trajectory
generator = ACEGenerator(
    agent=CodeAgent(model=model, tools=[]),
    playbook=playbook,
    use_step_callbacks=True,
)
logger.info("Generator: Execute + trajectory capture")

# Reflector: Analyze execution
reflector = ACEReflector(model=model)
logger.info("Reflector: Analyze + extract insights")

# Curator: Update playbook with dedup/pruning
curator = ACECurator(playbook=playbook, similarity_threshold=0.85, prune_threshold=-3)
logger.info("Curator: Deduplicate + prune + update playbook")

# ============================================================
# 6. STATISTICS & INTROSPECTION
# ============================================================
logger.info("=" * 70)
logger.info("6. STATISTICS & INTROSPECTION")
logger.info("=" * 70)

logger.info("Introspection methods:")
logger.debug("  - ace.stats()")
logger.debug("  - ace.show_playbook()")
logger.debug("  - ace.get_last_reflection()")
logger.debug("  - ace.get_last_curation_stats()")

logger.info("Example stats:")
logger.debug(f"Runs: {ace.run_count}")
logger.debug(f"Improvements: {ace.improvement_count}")
logger.debug(f"Playbook entries: {len(ace.playbook)}")
logger.debug(f"Version: {ace.playbook.version}")

# ============================================================
# 7. SHARED PLAYBOOK - Team Learning
# ============================================================
logger.info("=" * 70)
logger.info("7. SHARED PLAYBOOK")
logger.info("=" * 70)

# Multiple agents share the same playbook = team learning
shared_playbook = Playbook(
    name="team_knowledge",
    description="Shared across the team"
)

agent1 = ACEAgent(
    CodeAgent(model=model, tools=[]),
    playbook=shared_playbook,
    name="agent1",
    auto_improve=True,
)

agent2 = ACEAgent(
    ToolCallingAgent(model=model, tools=[]),
    playbook=shared_playbook,
    name="agent2",
    auto_improve=True,
)

logger.info(f"Shared playbook: '{shared_playbook.name}'")
logger.debug(f"Agent1: {agent1.name}")
logger.debug(f"Agent2: {agent2.name}")
logger.debug(f"Same playbook: {id(agent1.playbook) == id(agent2.playbook)}")
logger.debug("Team learning enabled!")

# ============================================================
# SUMMARY
# ============================================================
logger.info("=" * 70)
logger.info("✨ ACE: Self-Improving Agents")
logger.info("=" * 70)
logger.info("""
ACE = Generator → Reflector → Curator

Benefits:
✓ Pure orchestration (no wrappers)
✓ Agent-agnostic (works with any agent type)
✓ Knowledge portability (save/load playbooks)
✓ Team learning (shared playbooks)
✓ Production-ready (semantic dedup, pruning)

See README_ACE.md for more details!
""")
logger.info("=" * 70)
