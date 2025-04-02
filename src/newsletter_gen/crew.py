from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.newsletter_gen.tools.custom_tool import SearchAndContents, FindSimilar, GetContents
from datetime import datetime
import json
from langchain_groq import ChatGroq
from crewai import LLM
from langchain_core.agents import AgentFinish
import streamlit as st
from typing import Union, List, Tuple, Dict

@CrewBase
class NewsletterGen():
    """NewsletterGen crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def llm(self):
        llm = LLM(model="openai/gpt-4o-mini")
        return llm

    def step_callback(
        self,
        agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish],
        agent_name,
        *args,
    ):
        with st.chat_message("AI"):
            # Debugging: Display raw agent_output
            st.write("#### Debug: Raw Agent Output")
            st.code(str(agent_output))

            # Handle AgentAction objects (to capture the thought process)
            if hasattr(agent_output, "thought"):
                st.write(f"### Agent Name: {agent_name}")
                st.write("#### Thought Process:")
                st.write(agent_output.thought)  # Display the thought process immediately
                return  # Exit here since this is just the thought process

            # Handle list of actions with descriptions
            if isinstance(agent_output, list) and all(
                isinstance(item, tuple) for item in agent_output
            ):
                for action, description in agent_output:
                    st.write(f"### Agent Name: {agent_name}")
                    st.write(f"#### Tool used: {getattr(action, 'tool', 'Unknown')}")
                    st.write("#### Tool input:")
                    st.code(json.dumps(getattr(action, 'tool_input', 'Unknown'), indent=4), language="json")
                    st.write("#### Thought:")
                    st.write(getattr(action, 'log', 'Unknown'))
                    with st.expander("Show observation"):
                        st.markdown(f"**Observation:**\n\n{description}")

            # Handle AgentFinish output
            elif isinstance(agent_output, AgentFinish):
                st.write(f"### Agent Name: {agent_name}")
                st.write("#### Final Output:")
                output = agent_output.return_values.get("output", "No output provided")
                st.markdown(output, unsafe_allow_html=True)

            # Handle ToolResult objects
            elif hasattr(agent_output, "result"):
                st.write(f"### Agent Name: {agent_name}")
                st.write("#### Tool Result:")
                result = getattr(agent_output, "result", "No result provided")
                if isinstance(result, str):
                    st.code(result)
                elif isinstance(result, dict):
                    for key, value in result.items():
                        st.write(f"**{key.capitalize()}:** {value}")
                else:
                    st.write(result)

            # Handle unexpected formats
            elif isinstance(agent_output, dict):
                st.write(f"### Agent Name: {agent_name}")
                st.write("#### Output:")
                for key, value in agent_output.items():
                    st.write(f"**{key.capitalize()}:** {value}")

            else:
                st.write(f"### Agent Name: {agent_name}")
                st.write("#### Raw Output:")
                st.code(str(agent_output))

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[SearchAndContents(), FindSimilar(), GetContents()],
            verbose=True,
            llm=self.llm(),
            step_callback=lambda step: self.step_callback(step, "Research Agent"),
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config["editor"],
            verbose=True,
            tools=[SearchAndContents(), FindSimilar(), GetContents()],
            llm=self.llm(),
            step_callback=lambda step: self.step_callback(step, "Chief Editor"),
        )

    @agent
    def designer(self) -> Agent:
        return Agent(
            config=self.agents_config["designer"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm(),
            step_callback=lambda step: self.step_callback(step, "HTML Writer"),
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_research_task.md",
        )

    @task
    def edit_task(self) -> Task:
        return Task(
            config=self.tasks_config["edit_task"],
            agent=self.editor(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_edit_task.md",
        )

    @task
    def newsletter_task(self) -> Task:
        return Task(
            config=self.tasks_config["newsletter_task"],
            agent=self.designer(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_newsletter_task.html",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the NewsletterGen crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

