from llama_index.core.agent.workflow import (AgentOutput, AgentStream,
                                             AgentWorkflow,
                                             ToolCall, ToolCallResult)

from graphrag.agents.potential_collaborators_agent import potential_collaborators_agent as pot_col_agent
from graphrag.agents.projects_participants_agent import projects_participants_agent as pr_agent
from graphrag.agents.potential_consortium_organisations_agent import potential_consortium_organisations_agent as pot_con_org_agent


async def execute_agent_workflow(user_msg: str):
    agent_workflow = AgentWorkflow(
        agents=[pr_agent, pot_col_agent, pot_con_org_agent],
        root_agent=pr_agent.name,
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
