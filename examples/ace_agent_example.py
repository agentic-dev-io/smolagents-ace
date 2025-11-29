#!/usr/bin/env python
# coding=utf-8
"""
ACE (Agentic Context Engineering) Example

Demonstrates self-improving agents through evolving playbooks.
The agent accumulates knowledge across multiple tasks and improves over time.
"""

from smolagents import HfApiModel
from smolagents.ace import ACEAgent, Playbook, get_ace_tools

# Initialize model
model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct")

# Create ACE agent with auto-improvement enabled
agent = ACEAgent(
    model=model,
    tools=get_ace_tools(),  # Optional: Add ACE display tools
    auto_improve=True,
)

print("=" * 60)
print("ACE Agent Example - Self-Improving Agent")
print("=" * 60)

# Run multiple related tasks - the agent learns from each one
tasks = [
    "Calculate compound interest for $1000 at 5% annual rate over 10 years",
    "What's the NPV of cash flows [100, 200, 300, 400] with 8% discount rate?",
    "Calculate ROI if I invest $500 and get back $750",
]

for i, task in enumerate(tasks, 1):
    print(f"\n{'='*60}")
    print(f"Task {i}: {task}")
    print("=" * 60)

    result = agent.run(task)
    print(f"\nResult: {result}")

    # Show playbook growth
    stats = agent.stats()
    print(f"\nPlaybook: {stats['playbook']['total_entries']} entries, v{stats['playbook']['version']}")

# Show final playbook
print("\n" + "=" * 60)
print("FINAL PLAYBOOK")
print("=" * 60)
print(agent.show_playbook())

# Save playbook for future sessions
agent.save_playbook("finance_playbook.json")
print("\nPlaybook saved to finance_playbook.json")

# Show statistics
print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)
print(f"Total runs: {agent.run_count}")
print(f"Improvements: {agent.improvement_count}")
print(f"Playbook entries: {len(agent.playbook)}")

# Example: Resume with existing playbook
print("\n" + "=" * 60)
print("RESUMING WITH SAVED PLAYBOOK")
print("=" * 60)

# Create new agent with saved playbook
agent2 = ACEAgent.from_playbook(
    "finance_playbook.json",
    model=model,
    auto_improve=True,
)

# New task benefits from accumulated knowledge
result = agent2.run("Calculate the present value of $1000 received in 5 years at 6% rate")
print(f"Result: {result}")
print(f"Playbook now has {len(agent2.playbook)} entries")
