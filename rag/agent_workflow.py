from llama_index.core.agent.workflow import (AgentOutput, AgentStream,
                                             AgentWorkflow,
                                             ToolCall, ToolCallResult)

from rag.agents.sparql.potential_collaborators_agent import potential_collaborators_agent as pot_col_sp_agent
from rag.agents.sparql.projects_participants_agent import projects_participants_agent as pr_sp_agent
from rag.agents.cypher.potential_collaborators_agent import potential_collaborators_agent as pot_col_cy_agent
from rag.agents.cypher.projects_participants_agent import projects_participants_agent as pr_cy_agent


async def execute_sparql_agent_workflow(user_msg: str):
    agent_workflow = AgentWorkflow(
        agents=[pr_sp_agent, pot_col_sp_agent],
        root_agent=pr_sp_agent.name,
        initial_state={
        },
    )
    handler = agent_workflow.run(
        user_msg=user_msg,
    )

    current_agent = None
    async for event in handler.stream_events():
        if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'=' * 50}")
            print(f"Agent: {current_agent}")
            print(f"{'=' * 50}\n")

        if isinstance(event, AgentStream):
            if event.delta:
                print(event.delta, end="", flush=True)
        # elif isinstance(event, AgentInput):
        #     print("Input:", event.input)
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("Output:", event.response.content)
            if event.tool_calls:
                print(
                    "Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"Tool Result ({event.tool_name}):")
            # print(f"  Arguments: {event.tool_kwargs}")
            # print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling Tool: {event.tool_name}")
            # print(f"  With arguments: {event.tool_kwargs}")
    response = await handler

    return response.response.content


async def execute_cypher_agent_workflow(user_msg: str):
    agent_workflow = AgentWorkflow(
        agents=[pr_cy_agent, pot_col_cy_agent],
        root_agent=pr_cy_agent.name,
        initial_state={
        },
    )
    handler = agent_workflow.run(
        user_msg=user_msg,
    )

    current_agent = None
    async for event in handler.stream_events():
        if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'=' * 50}")
            print(f"Agent: {current_agent}")
            print(f"{'=' * 50}\n")

        if isinstance(event, AgentStream):
            if event.delta:
                print(event.delta, end="", flush=True)
        # elif isinstance(event, AgentInput):
        #     print("Input:", event.input)
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("Output:", event.response.content)
            if event.tool_calls:
                print(
                    "Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"Tool Result ({event.tool_name}):")
            # print(f"  Arguments: {event.tool_kwargs}")
            # print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling Tool: {event.tool_name}")
            # print(f"  With arguments: {event.tool_kwargs}")
    response = await handler

    return response.response.content
