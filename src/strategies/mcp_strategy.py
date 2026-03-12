import json
import h11
import logging

from typing import Optional

from util.tools import is_azure_environment
from azure.identity import get_bearer_token_provider

from opentelemetry.trace import (
    SpanKind,
    format_span_id,
    format_trace_id,
    get_current_span
)

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.mcp import MCPSsePlugin

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies

from connectors.appconfig import AppConfigClient
from dependencies import get_config
from telemetry import Telemetry

tracer = Telemetry.get_tracer(__name__)

class McpStrategy(BaseAgentStrategy):
    """
    Implements a MCP Server strategy. This class handles creating an agent, sending
    a user message, streaming the response, and cleaning up resources.
    """

    #override this method in subclasses to provide custom headers for user identity
    def write_headers(self, headers, write) -> None:
        # "Since the Host field-value is critical information for handling a
        # request, a user agent SHOULD generate Host as the first header field
        # following the request-line." - RFC 7230
        raw_items = headers._full_items
        for raw_name, name, value in raw_items:
            if name == b"host":
                write(b"%s: %s\r\n" % (raw_name, value))
        for raw_name, name, value in raw_items:
            if name != b"host":
                write(b"%s: %s\r\n" % (raw_name, value))

        #write the user context header if it exists
        if self.user_context:
            write(b"user-context: %s\r\n" % json.dumps(self.user_context).encode('utf-8'))
        write(b"\r\n")

    def __init__(self):
        """
        Initialize base credentials and tools.
        """
        super().__init__()

        # Force all logs at DEBUG or above to appear
        logging.debug("Initializing McpStrategy...")

        # Strategy type identifier
        self.strategy_type = AgentStrategies.MCP

        cfg = get_config()

        # Agent Tools Initialization Section
        # =========================================================

        # Allow the user to specify an existing agent ID (optional)
        self.existing_agent_id = cfg.get("AGENT_ID", "") or None

        # need to inject the user_context header into the h11 writer
        # this is a workaround for the fact that h11 does not support custom headers
        h11._writers.write_headers = self.write_headers
    
    async def create():
        cfg = get_config()
        instance = McpStrategy()
        instance.config = cfg
        instance.kernel = Kernel()

        # Agent Tools Initialization Section
        # =========================================================
        instance.tools_list = []
        instance.tool_resources = {}
        
        logging.debug(f"Final tools_list: {instance.tools_list}")
        logging.debug(f"Final tool_resources: {instance.tool_resources}")

        token_provider = get_bearer_token_provider(
            cfg.credential,
            "https://ai.azure.com/.default"
        )

        instance.model = instance._get_model()

        instance.kernel.add_service(AzureChatCompletion(
            service_id=instance.model.get("name"),
            deployment_name=instance.model.get("name"),
            endpoint=instance.model.get("endpoint"),
            api_version=instance.model.get("version"),
            ad_token_provider= token_provider
        ))

        instance.agent = ChatCompletionAgent(
            kernel=instance.kernel,
            #function_choice_behavior=FunctionChoiceBehavior.AUTO,
            name="MultiPluginAgent",
            #plugins=instance.tools_list
        )

        return instance
    
    async def _create_mcp_plugin(self, extra_headers: Optional[dict] = {}) -> MCPSsePlugin:
        # Add an MCP Server tool if configured
        mcp_server_url = self.cfg.get("MCP_APP_ENDPOINT", default="http://localhost:80") + "/sse"

        if not is_azure_environment():
            mcp_server_url = 'http://localhost:5000/sse'

        mcp_server_timeout = self.cfg.get("MCP_CLIENT_TIMEOUT", default=600, type=int)
        mcp_server_api_key = self.cfg.get("MCP_APP_APIKEY", default=None)
        mcp_server_transport = self.cfg.get("MCP_SERVER_TRANSPORT", default="sse")

        if mcp_server_url:
            logging.debug(f"Adding MCP Server tool with URL: {mcp_server_url}")

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                'user-context': json.dumps(self.user_context) if self.user_context else '{}'
            }

            # Add any extra headers provided
            if extra_headers:
                logging.debug(f"Adding extra headers: {extra_headers}")
                headers.update(extra_headers)

            if mcp_server_api_key:
                logging.debug("MCP Server API key provided; adding to headers.")
                headers["X-API-KEY"] = f"{mcp_server_api_key}"

            if mcp_server_transport == "sse":
                # Use the MCPSsePlugin for SSE transport
                logging.debug("Using MCPSsePlugin for MCP Server with SSE transport.")
                plugin = MCPSsePlugin(
                    name="McpServerPlugin",
                    url=mcp_server_url,
                    headers=headers,
                    timeout=mcp_server_timeout,
                    kernel=self.kernel
                )

                await plugin.connect()

                return plugin

        else:
            logging.debug("No MCP Server URL provided; skipping MCP Server tool initialization.")


    async def initiate_agent_flow(self, user_message: str):
        """
        Sends a user message and yields streamed chunks.
        Manages thread creation/reuse entirely here, storing both
        agent_id, thread_id and full messages list in self.conversation.
        """
        logging.debug(f"invoke_stream called with user_message: {user_message!r}")
        conv = self.conversation
        thread_id = conv.get("thread_id")
        logging.debug(f"Current conversation state: thread_id={thread_id}")

        conv["agent_id"] = self.agent.id

        plugin = await self._create_mcp_plugin(extra_headers={})

        self.kernel.add_plugin(plugin)

        with tracer.start_as_current_span('initiate_agent_flow', kind=SpanKind.CLIENT) as span:
        
            response = await self.agent.get_response(messages=user_message)

            yield response
        
            conv["messages"] = []
            conv["messages"].append({
                "role": "system",
                "text": response.message.content
            })
            conv['completion_tokens'] = response.metadata.get('usage').completion_tokens
            conv['prompt_tokens'] = response.metadata.get('usage').prompt_tokens

            if self.user_context:
                conv['user_context'] = self.user_context

            logging.debug(f"Final conversation messages: {conv['messages']}")
