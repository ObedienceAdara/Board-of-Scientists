"""Graph node boundary.

Each named node is imported from the legacy implementation temporarily. This
keeps the LangGraph topology stable while responsibility moves into the new
module layout.
"""
from board_of_scientists._legacy.agents import (
    analyst_agent, cro_read_paper, cro_create_plan, cro_evaluate_agent,
    cro_final_verdict, theorist_agent, architect_agent, engineer_agent,
    reviewer_agent, experiment_engineer_agent, writer_agent,
)
node_analyst = analyst_agent
node_cro_read = cro_read_paper
node_theorist = theorist_agent
node_architect = architect_agent
node_cro_plan = cro_create_plan
node_engineer = engineer_agent
node_reviewer = reviewer_agent
node_experiment = experiment_engineer_agent
node_writer = writer_agent
node_cro_verdict = cro_final_verdict
