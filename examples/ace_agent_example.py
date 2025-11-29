#!/usr/bin/env python
# coding=utf-8
"""
ACE (Agentic Context Engineering) Example

Demonstrates self-improving agents through evolving playbooks.
ACE orchestrates existing smolagents (CodeAgent, ToolCallingAgent) - no wrappers.
"""

from smolagents import CodeAgent, ToolCallingAgent, HfApiModel
from smolagents.ace import ACEAgent, ACEGenerator, Playbook

# Initialize model
model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct")

# Create any smolagents agent
agent = CodeAgent(model=model, tools=[])
# Or: agent = ToolCallingAgent(model=model, tools=[])

# ACE orchestrates the agent - no inheritance, no wrappers
ace = ACEAgent(agent, auto_improve=True)

print("=" * 60)
print("ACE Example - Self-Improving Agent")
print(f"Using: {type(ace.agent).__name__}")  # Access underlying agent
print("=" * 60)

# Run multiple related tasks - Playbook grows automatically
tasks = [
    "Calculate compound interest for $1000 at 5% annual rate over 10 years",
    "What's the NPV of cash flows [100, 200, 300, 400] with 8% discount rate?",
    "Calculate ROI if I invest $500 and get back $750",
]

for i, task in enumerate(tasks, 1):
    print(f"\n{'='*60}")
    print(f"Task {i}: {task}")
    print("=" * 60)

    result = ace.run(task)
    print(f"\nResult: {result}")

    # Show playbook growth
    stats = ace.stats()
    print(f"\nPlaybook: {stats['playbook']['total_entries']} entries, v{stats['playbook']['version']}")

# Show final playbook
print("\n" + "=" * 60)
print("FINAL PLAYBOOK")
print("=" * 60)
print(ace.show_playbook())

# Save playbook for future sessions
ace.save_playbook("finance_playbook.json")
print("\nPlaybook saved to finance_playbook.json")

# Statistics
print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)
print(f"Total runs: {ace.run_count}")
print(f"Improvements: {ace.improvement_count}")
print(f"Playbook entries: {len(ace.playbook)}")

# ============================================================
# Advanced: Use roles individually
# ============================================================
print("\n" + "=" * 60)
print("ADVANCED: Individual Roles")
print("=" * 60)

# Generator alone - just playbook injection + trajectory extraction
generator = ACEGenerator(agent=CodeAgent(model=model, tools=[]))
gen_result = generator.run("Calculate 2^10")
print(f"Generator result: {gen_result['result']}")
print(f"Trajectory steps: {len(gen_result['trajectory'])}")

# ============================================================
# Resume with saved playbook
# ============================================================
print("\n" + "=" * 60)
print("RESUME WITH SAVED PLAYBOOK")
print("=" * 60)

# Load playbook and create new ACE orchestrator
playbook = Playbook.load("finance_playbook.json")
new_agent = ToolCallingAgent(model=model, tools=[])  # Different agent type!
ace2 = ACEAgent(new_agent, playbook=playbook, auto_improve=True)

print(f"Loaded playbook v{playbook.version} with {len(playbook)} entries")
print(f"Now using: {type(ace2.agent).__name__}")

# New task benefits from accumulated knowledge
result = ace2.run("Calculate present value of $1000 received in 5 years at 6%")
print(f"Result: {result}")
