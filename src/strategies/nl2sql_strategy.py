import logging
import re
from typing import AsyncIterator

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents.orchestration.group_chat import GroupChatOrchestration, RoundRobinGroupChatManager
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from azure.identity import get_bearer_token_provider

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from plugins.nl2sql.plugin import NL2SQLPlugin


class NL2SQLStrategy(BaseAgentStrategy):
    """
    An optimized NL2SQL Retrieval-Augmented Generation strategy
    """
    def __init__(self):
        super().__init__()
        self.strategy_type = AgentStrategies.NL2SQL

        # single plugin instance
        self._nl2sql_plugin = NL2SQLPlugin()

        # precompile the terminator-cleanup regex
        self._terminator_re = re.compile(r'\bterminate\b', re.IGNORECASE)

        # placeholders for prompts (lazy-loaded)
        self._triage_prompt    = None
        self._sqlquery_prompt  = None

    async def _load_prompts(self):
        """Load and cache the three prompt templates once per instance."""
        if self._triage_prompt is None:
            self._triage_prompt      = await self._read_prompt("triage_agent")
            self._sqlquery_prompt    = await self._read_prompt("sqlquery_agent")
            self._syntetizer_prompt  = await self._read_prompt("syntetizer_agent")

    async def initiate_agent_flow(self, user_message: str) -> AsyncIterator[str]:
        # ensure prompts are loaded
        await self._load_prompts()

        # create kernel and AI service
        kernel = Kernel()
        token_provider = get_bearer_token_provider(
            self.credential, "https://cognitiveservices.azure.com/.default"
        )
        service = AzureChatCompletion(
            deployment_name=self.model_name,
            endpoint=self.account_endpoint,
            api_version=self.openai_api_version,
            ad_token_provider=token_provider,
        )
        kernel.add_service(service)

        # create local agents (no server-side creation needed)
        triage_agent = ChatCompletionAgent(
            kernel=kernel,
            name="TriageAgent",
            description="Triages NL2SQL queries and routes to the appropriate agent.",
            instructions=self._triage_prompt,
            plugins=[self._nl2sql_plugin],
        )
        sqlquery_agent = ChatCompletionAgent(
            kernel=kernel,
            name="SQLQueryAgent",
            description="Generates and executes SQL queries based on natural language input.",
            instructions=self._sqlquery_prompt,
            plugins=[self._nl2sql_plugin],
        )
        syntetizer_agent = ChatCompletionAgent(
            kernel=kernel,
            name="SyntetizerAgent",
            description="Synthesizes results from SQL queries into a natural language response.",
            instructions=self._syntetizer_prompt,
            plugins=[self._nl2sql_plugin],
        )

        runtime = InProcessRuntime()
        runtime.start()

        try:
            orchestration = GroupChatOrchestration(
                members=[triage_agent, sqlquery_agent, syntetizer_agent],
                manager=RoundRobinGroupChatManager(max_rounds=10),
            )

            result = await orchestration.invoke(
                task=user_message,
                runtime=runtime,
            )

            # process result — strip terminator keyword and yield
            final_text = str(result)
            cleaned = self._terminator_re.sub("", final_text)
            yield cleaned

        finally:
            try:
                await runtime.stop_when_idle()
            except Exception as e:
                logging.warning(f"Runtime stop failed: {e!r}")
