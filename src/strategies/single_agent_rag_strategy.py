import logging
import json
from typing import Any, Optional

from azure.ai.projects.models import (
    PromptAgentDefinition,
    AzureAISearchTool,
    AzureAISearchToolResource,
    AISearchIndexResource,
    BingGroundingTool,
    BingGroundingSearchToolParameters,
    BingGroundingSearchConfiguration,
    FunctionTool,
)
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent
)
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies

from connectors.appconfig import AppConfigClient
from dependencies import get_config

class SingleAgentRAGStrategy(BaseAgentStrategy):
    """
    Implements a single-agent Retrieval-Augmented Generation (RAG) strategy
    using Azure AI Foundry. This class handles creating an agent, sending
    a user message, streaming the response, and cleaning up resources.
    """

    def __init__(self):
        """
        Initialize base credentials and tools.
        """
        super().__init__()

        # Force all logs at DEBUG or above to appear
        logging.debug("Initializing SingleAgentRAGStrategy...")

        # Strategy type
        self.strategy_type = AgentStrategies.SINGLE_AGENT_RAG

        cfg = get_config()

        # Agent Tools Initialization Section
        # =========================================================

        # Allow the user to specify an existing agent ID (optional)
        # Use a safe default to avoid raising when key is not present
        self.existing_agent_id = cfg.get("AGENT_ID", "") or None

        # Initialize tool containers
        self.tools_list = []
        self.tool_resources = {}

        # --- Load Agentic Retrieval Configuration ---
        self.enable_agentic_retrieval = cfg.get("ENABLE_AGENTIC_RETRIEVAL", "false", str).lower() == "true"
        logging.debug(f"Agentic Retrieval Enabled: {self.enable_agentic_retrieval}")

        self.agentic_retrieval_client = None
        self._recent_thread_messages = {}
        self._last_user_message_by_thread = {}
        self.agentic_retrieval_credential = None

        if self.enable_agentic_retrieval:
            self.search_query_endpoint = cfg.get("SEARCH_SERVICE_QUERY_ENDPOINT", "")
            self.knowledge_agent_name = cfg.get("SEARCH_SERVICE_AGENT_NAME", "")
            if not self.knowledge_agent_name:
                resource_token = cfg.get("RESOURCE_TOKEN", "")
                if resource_token:
                    self.knowledge_agent_name = f"ragindex-{resource_token}-rag-agent"
                else:
                    logging.warning("RESOURCE_TOKEN not found. Cannot construct default knowledge agent name.")
            logging.debug(f"Knowledge Agent Name: {self.knowledge_agent_name}")
            logging.debug(f"Search Query Endpoint: {self.search_query_endpoint}")
            if not self.search_query_endpoint or not self.knowledge_agent_name:
                logging.error(
                    "Agentic retrieval is enabled but SEARCH_SERVICE_QUERY_ENDPOINT or agent name is not configured. Falling back to traditional search."
                )
                self.enable_agentic_retrieval = False
            else:
                try:
                    self.agentic_retrieval_credential = SyncDefaultAzureCredential(
                        exclude_interactive_browser_credential=True
                    )
                except Exception as cred_error:
                    logging.error(
                        "Unable to initialize synchronous credential for agentic retrieval: %s",
                        cred_error,
                        exc_info=True,
                    )
                    self.enable_agentic_retrieval = False
                    logging.error("Agentic retrieval disabled due to credential initialization failure")

        # --- Initialize BingGroundingTool (if configured) ---
        bing_conn = cfg.get("BING_CONNECTION_ID", "")
        if not bing_conn:
            logging.warning(
                "BING_CONNECTION_ID not set in App Config variables. "
                "BingGroundingTool will not be available."
            )
        else:
            bing = BingGroundingTool(
                bing_grounding=BingGroundingSearchToolParameters(
                    search_configurations=[
                        BingGroundingSearchConfiguration(project_connection_id=bing_conn)
                    ]
                )
            )
            self.tools_list.append(bing)
            logging.debug(f"Added BingGroundingTool to tools_list: {bing}")

        # --- Initialize AzureAISearchTool (only if agentic retrieval is disabled) ---
        if not self.enable_agentic_retrieval:
            azure_ai_conn_id = cfg.get("SEARCH_CONNECTION_ID", "")
            index_name = cfg.get("SEARCH_RAG_INDEX_NAME", "ragindex")
            logging.debug(f"seachConnectionId (cfg)  = {azure_ai_conn_id}")
            logging.debug(f"SEARCH_RAG_INDEX_NAME (cfg) = {index_name}")
            if not azure_ai_conn_id:
                logging.warning(
                    "seachConnectionId undefined (cfg). "
                    "AzureAISearchTool will be unavailable."
                )
            if not index_name:
                logging.warning(
                    "SEARCH_RAG_INDEX_NAME undefined (cfg). "
                    "AzureAISearchTool will be unavailable."
                )
            self.ai_search = AzureAISearchTool(
                azure_ai_search=AzureAISearchToolResource(
                    indexes=[AISearchIndexResource(
                        project_connection_id=azure_ai_conn_id,
                        index_name=index_name,
                        query_type="simple",
                    )]
                )
            )
            logging.debug(f"Created AzureAISearchTool: {self.ai_search}")
            self.tools_list.append(self.ai_search)
        else:
            logging.info("Using Agentic Retrieval - traditional AzureAISearchTool will not be initialized")

        logging.debug(f"Final tools_list: {self.tools_list}")
        logging.debug(f"Final tool_resources: {self.tool_resources}")

    def _create_agentic_retrieval_tool(self, conversation_id: str):
        if not self.enable_agentic_retrieval:
            logging.warning("Agentic retrieval is not enabled. Tool will not be created.")
            return None
        try:
            if not self.agentic_retrieval_credential:
                logging.error("Synchronous credential not available for agentic retrieval client initialization")
                return None

            self.agentic_retrieval_client = KnowledgeAgentRetrievalClient(
                endpoint=self.search_query_endpoint,
                agent_name=self.knowledge_agent_name,
                credential=self.agentic_retrieval_credential
            )
            logging.info(f"KnowledgeAgentRetrievalClient initialized for agent: {self.knowledge_agent_name}")
            
            def agentic_retrieval(query: Optional[str] = None) -> str:
                """
                Search and retrieve relevant information from the knowledge base to answer user questions.
                Use this function whenever you need to find information to answer the user's query.
                This function will return relevant documents with citations that you should use in your response.
                
                Returns:
                    str: JSON array of documents with ref_id, content, and metadata fields.
                """
                try:
                    logging.info(f"[agentic_retrieval] Function called for conversation: {conversation_id}")
                    
                    # Get cached messages for retrieval context
                    converted_messages = self._recent_thread_messages.get(conversation_id, [])
                    if not converted_messages:
                        logging.warning("[agentic_retrieval] No cached messages found; falling back to last user message")
                        fallback_text = query or self._last_user_message_by_thread.get(conversation_id, "")
                        if fallback_text:
                            converted_messages = [
                                KnowledgeAgentMessage(
                                    role="user",
                                    content=[KnowledgeAgentMessageTextContent(text=fallback_text)]
                                )
                            ]
                        else:
                            logging.error("[agentic_retrieval] Unable to determine request context for retrieval")
                            return json.dumps([{
                                "ref_id": 0,
                                "content": "Unable to determine the user query for retrieval.",
                                "title": "Retrieval Error"
                            }])
                    logging.debug(f"Converted {len(converted_messages)} messages for retrieval request (query override provided: {bool(query)})")
                    
                    logging.info(f"[agentic_retrieval] Calling retrieve with {len(converted_messages)} messages")
                    for idx, msg in enumerate(converted_messages):
                        logging.debug(f"[agentic_retrieval] Message {idx}: role={msg.role}, content_preview={msg.content[0].text[:100] if msg.content else 'empty'}...")
                    
                    retrieval_result = self.agentic_retrieval_client.retrieve(
                        retrieval_request=KnowledgeAgentRetrievalRequest(
                            messages=converted_messages
                        )
                    )
                    
                    logging.info("[agentic_retrieval] Retrieval completed successfully")
                    
                    if retrieval_result.response and len(retrieval_result.response) > 0:
                        response_content = retrieval_result.response[0].content
                        if response_content and len(response_content) > 0:
                            result_text = response_content[0].text
                            logging.debug(
                                "[agentic_retrieval] Raw retrieval payload (truncated): %s",
                                result_text[:200],
                            )

                            transformed_text = self._inject_citation_labels(result_text, retrieval_result)

                            logging.info(
                                "[agentic_retrieval] Returning %d characters of transformed content",
                                len(transformed_text),
                            )
                            if hasattr(retrieval_result, 'activity') and retrieval_result.activity:
                                logging.info(
                                    "[agentic_retrieval] Activity: %d operations logged",
                                    len(retrieval_result.activity),
                                )
                                for activity in retrieval_result.activity[:3]:
                                    activity_type = getattr(activity, 'type', 'unknown')
                                    logging.debug(f"Activity type: {activity_type}")
                            return transformed_text
                    
                    logging.warning("[agentic_retrieval] Empty response from retrieval service")
                    return json.dumps([
                        {
                            "ref_id": 0,
                            "content": "No information found.",
                            "title": "Empty Result",
                            "citation": "No Source",
                        }
                    ])
                except Exception as e:
                    logging.error(f"[agentic_retrieval] Error during retrieval: {str(e)}", exc_info=True)
                    return json.dumps([
                        {
                            "ref_id": 0,
                            "content": "Unable to retrieve information at this time. Please try again.",
                            "title": "Retrieval Error",
                            "citation": "Retrieval Error",
                        }
                    ])
            
            # Create schema-based FunctionTool for 2.0 API (no auto-execution)
            function_tool = FunctionTool(
                name="agentic_retrieval",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional query override for retrieval",
                        }
                    },
                    "required": [],
                },
                description="Search and retrieve relevant information from the knowledge base to answer user questions.",
            )
            
            logging.info("Agentic retrieval tool created successfully")
            # Return: (tool definition for agent, callable function for dispatch)
            return (function_tool, agentic_retrieval)
        except Exception as e:
            logging.error(f"Failed to create agentic retrieval tool: {str(e)}", exc_info=True)
            return None

    def _build_reference_lookup(self, retrieval_result) -> dict[str, str]:
        """Create a mapping from reference id to human-readable source label."""
        lookup: dict[str, str] = {}
        references = getattr(retrieval_result, "references", None) or []
        for ref in references:
            ref_id = None
            source_data = {}
            try:
                ref_dict = ref.as_dict()
                ref_id = ref_dict.get("id")
                source_data = ref_dict.get("source_data") or {}
            except AttributeError:
                ref_id = getattr(ref, "id", None)
                source_data = getattr(ref, "source_data", {}) or {}

            if ref_id is None:
                continue

            candidates = [
                source_data.get("filepath"),
                source_data.get("file_path"),
                source_data.get("metadata_storage_path"),
                source_data.get("metadata_storage_name"),
                source_data.get("file_name"),
                source_data.get("fileName"),
                source_data.get("citation"),
                source_data.get("displayName"),
                source_data.get("display_name"),
                source_data.get("title"),
                source_data.get("name"),
                source_data.get("id"),
                source_data.get("uri"),
            ]
            label = self._choose_label(candidates)
            if not label:
                label = str(ref_id)

            lookup[str(ref_id)] = label

        return lookup

    @staticmethod
    def _choose_label(candidates: list[Any]) -> Optional[str]:
        """Select the best citation label, preferring values that include a file extension."""
        cleaned = []
        for value in candidates:
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    cleaned.append(candidate)

        for candidate in cleaned:
            if "." in candidate:
                return candidate

        return cleaned[0] if cleaned else None

    def _inject_citation_labels(self, result_text: str, retrieval_result) -> str:
        """Attach human-readable citation labels to retrieval payload."""
        try:
            payload = json.loads(result_text)
        except json.JSONDecodeError:
            logging.debug("[agentic_retrieval] Retrieval response is not JSON; skipping citation augmentation")
            return result_text

        if not isinstance(payload, list):
            logging.debug("[agentic_retrieval] Retrieval JSON is not a list; skipping citation augmentation")
            return result_text

        reference_lookup = self._build_reference_lookup(retrieval_result)

        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue

            metadata = item.get("metadata") or {}
            
            # Get filepath candidates
            filepath_candidates = [
                item.get("filepath"),
                item.get("file_path"),
                item.get("file_name"),
                item.get("fileName"),
                metadata.get("filepath"),
                metadata.get("file_path"),
                metadata.get("metadata_storage_name"),
                metadata.get("metadata_storage_path"),
                metadata.get("file_name"),
                metadata.get("fileName"),
            ]
            
            # Get title candidates
            title_candidates = [
                item.get("title"),
                item.get("citation"),
                metadata.get("displayName"),
                metadata.get("display_name"),
                metadata.get("title"),
                metadata.get("name"),
            ]

            ref_identifier = item.get("ref_id")
            if ref_identifier is not None:
                ref_label = reference_lookup.get(str(ref_identifier))
                if ref_label:
                    # Check if ref_label looks like a filepath (has extension)
                    if "." in ref_label:
                        filepath_candidates.insert(0, ref_label)
                    else:
                        title_candidates.insert(0, ref_label)

            filepath = self._choose_label(filepath_candidates)
            title = self._choose_label(title_candidates)

            if not filepath:
                if isinstance(ref_identifier, str) and ref_identifier.strip():
                    filepath = ref_identifier.strip()
                elif isinstance(ref_identifier, int):
                    filepath = f"Document_{ref_identifier}.pdf"
                else:
                    filepath = f"Document_{idx}.pdf"

            if not title:
                if filepath and "." in filepath:
                    # Extract title from filepath by removing extension and replacing underscores
                    title = filepath.rsplit(".", 1)[0].replace("_", " ")
                else:
                    title = f"Document {idx}"

            # Format as [title](filepath)
            item["citation"] = f"[{title}]({filepath})"

        return json.dumps(payload)

    async def _capture_recent_conversation_messages(self, openai_client, conversation_id: str, limit: int = 5):
        """Collect the most recent conversational messages for retrieval context."""
        try:
            items = await openai_client.conversations.items.list(conversation_id=conversation_id)
            collected = []
            async for item in items:
                collected.append(item)
            # Take only the last `limit` items
            collected = collected[-limit:]

            converted_messages = []
            for msg in collected:
                role = getattr(msg, "role", None)
                if role == "system":
                    continue
                content_text = ""
                # Extract text from item content
                content_list = getattr(msg, "content", None)
                if content_list:
                    for content_item in content_list:
                        text_attr = getattr(content_item, "text", None)
                        if text_attr:
                            content_text = text_attr if isinstance(text_attr, str) else getattr(text_attr, "value", str(text_attr))
                            break
                if not content_text:
                    # Try direct text attribute 
                    content_text = getattr(msg, "text", "") or ""
                if content_text and role:
                    converted_messages.append(
                        KnowledgeAgentMessage(
                            role=role,
                            content=[KnowledgeAgentMessageTextContent(text=content_text)]
                        )
                    )
            logging.debug(
                "Prepared %d messages for agentic retrieval (conversation_id=%s)",
                len(converted_messages),
                conversation_id,
            )
            return converted_messages
        except Exception as exc:
            logging.error(
                "Failed to capture recent messages for conversation %s: %s",
                conversation_id,
                str(exc),
                exc_info=True,
            )
            return []

    async def create():
        """
        Factory method to create an instance of SingleAgentRAGStrategy.
        Initializes the agent and tools.
        """
        logging.debug("Creating SingleAgentRAGStrategy instance...")
        instance = SingleAgentRAGStrategy()

        return instance


    async def initiate_agent_flow(self, user_message: str):
        """
        Initiates the agent flow using azure-ai-projects 2.0 conversation/responses API.
        
        Uses two clients:
        - project_client: Agent CRUD (create_version, delete_version)
        - openai_client: Conversations and responses (streaming)
        
        1. AGENTIC RETRIEVAL ENABLED (self.enable_agentic_retrieval = True):
           - Uses KnowledgeAgentRetrievalClient for intelligent retrieval
           - Creates schema-based FunctionTool for agentic_retrieval
           - Manual function-call dispatch loop (no auto-execution in 2.0)
           - Traditional AzureAISearchTool is NOT initialized
        
        2. AGENTIC RETRIEVAL DISABLED (self.enable_agentic_retrieval = False):
           - Uses traditional AzureAISearchTool
           - Tool is configured during __init__ and added to tools_list
           - Agent uses built-in AzureAISearchTool capabilities
        
        Both modes can optionally use BingGroundingTool if BING_CONNECTION_ID is configured.
        """
        logging.debug(f"invoke_stream called with user_message: {user_message!r}")
        conv = self.conversation
        # Backward-compatible conversation ID read (thread_id → conversation_id)
        conversation_id = conv.get("conversation_id") or conv.get("thread_id")
        logging.debug(f"Current conversation state: conversation_id={conversation_id}")

        async with self.project_client as project_client:
            openai_client = project_client.get_openai_client()

            # Setup agentic retrieval tool if enabled (BEFORE creating agent)
            agentic_tool_definition = None
            agentic_function = None
            if self.enable_agentic_retrieval:
                logging.info("Setting up agentic retrieval tool...")
                result = self._create_agentic_retrieval_tool(conversation_id or "new")
                if result:
                    agentic_tool_definition, agentic_function = result
                    logging.info(f"Agentic retrieval function registered for manual dispatch")
                else:
                    logging.warning("Failed to create agentic retrieval functions, proceeding without it")

            # Agent management
            create_agent = False
            agent = None
            if self.existing_agent_id:
                logging.debug("agent_id exists; retrieving existing agent...")
                agent = await project_client.agents.get_agent(self.existing_agent_id)
                logging.info(f"Reused agent with ID: {agent.id}")
            else:
                logging.debug("creating agent via create_version()...")
                instructions = await self._read_prompt("main")
                instructions += """

UNIVERSAL CITATION REQUIREMENTS:

1. Every fact derived from retrieved content MUST include a citation using the format [title](filepath).
2. The title should be descriptive (e.g., document title or display name) and filepath should be the actual filename with extension.
3. Place the citation immediately after the sentence that uses the sourced information.
4. Repeat citations when the same source backs multiple statements.
5. If no relevant document is found, respond with "I don't have information about that in my knowledge base."
"""

                if self.enable_agentic_retrieval:
                    instructions += """

CRITICAL INSTRUCTIONS FOR AGENTIC RETRIEVAL:

1. You MUST call the 'agentic_retrieval' function to search for information BEFORE answering ANY question.
2. Do NOT attempt to answer from your internal knowledge without calling the function first.
3. The function returns a JSON array containing 'content', 'metadata', and a 'citation' field formatted as [title](filepath) for each source.
4. You MUST cite ALL sources using the exact format [title](filepath) as provided in the citation field (for example, [Northwind Health Plus Benefits Details](Northwind_Health_Plus_Benefits_Details.pdf)).
5. Place citations immediately after each statement that uses information from the sources.
6. If multiple statements use the same source, repeat the citation after each relevant statement.
7. If the function returns no relevant documents or empty results, respond with "I don't have information about that in my knowledge base."

Example of proper response:
Question: "What is the emergency room copay?"
After calling agentic_retrieval and receiving a document with citation "[Benefits Summary](benefits-summary.pdf)":
"The emergency room copay for in-network services is $100 [Benefits Summary](benefits-summary.pdf). For out-of-network services, the copay is $150 [Benefits Summary](benefits-summary.pdf)."

REMEMBER: Call agentic_retrieval FIRST, ALWAYS cite your sources using the provided citation labels, and NEVER answer without using the function."""
                else:
                    instructions += """

GUIDANCE FOR AZURE AI SEARCH TOOL:

1. Invoke the Azure AI Search tool to gather grounding data before answering.
2. The search results will include citation annotations that you must use in the format [title](filepath).
3. Use the exact citation format provided by the tool's annotations.
4. If the metadata lacks a descriptive name, construct a concise label and use it consistently in citations.
5. Never rely solely on internal knowledge when the search results contain relevant information.

Example: "The emergency room copay for in-network services is $100 [Benefits Summary](benefits-summary.pdf)."""
                
                # Prepare tools list — add agentic retrieval tool if enabled
                tools_list = self.tools_list.copy()
                if self.enable_agentic_retrieval and agentic_tool_definition:
                    tools_list.append(agentic_tool_definition)
                    logging.info("Added agentic_retrieval to agent tools list")
                
                agent = await project_client.agents.create_version(
                    agent_name="gpt-rag-agent",
                    definition=PromptAgentDefinition(
                        model=self.model_name,
                        instructions=instructions,
                        tools=tools_list,
                        tool_resources=self.tool_resources,
                    ),
                )
                create_agent = True
                logging.info(f"Created agent: name={agent.name}, version={agent.version}")

            conv["agent_name"] = agent.name
            conv["agent_version"] = agent.version

            # Conversation management — create or reuse
            if not conversation_id:
                logging.debug("No conversation_id; creating new conversation with user message...")
                conversation = await openai_client.conversations.create(
                    items=[{"type": "message", "role": "user", "content": user_message}],
                )
                conversation_id = conversation.id
                logging.info(f"Created new conversation with ID: {conversation_id}")
            else:
                logging.debug(f"Reusing conversation_id={conversation_id}; adding user message...")
                await openai_client.conversations.items.create(
                    conversation_id=conversation_id,
                    item={"type": "message", "role": "user", "content": user_message},
                )
                logging.info(f"Added user message to existing conversation: {conversation_id}")

            conv["conversation_id"] = conversation_id
            logging.debug(f"Stored conv['conversation_id'] = {conversation_id}")

            # Cache messages for agentic retrieval
            if self.enable_agentic_retrieval:
                self._last_user_message_by_thread[conversation_id] = user_message
                cached_messages = await self._capture_recent_conversation_messages(
                    openai_client, conversation_id
                )
                if cached_messages:
                    self._recent_thread_messages[conversation_id] = cached_messages
                else:
                    logging.warning(
                        "Agentic retrieval cache is empty for conversation %s; "
                        "retrieval will rely on tool arguments",
                        conversation_id,
                    )

            # Stream response using the 2.0 responses API with SSE events
            logging.debug(
                f"Streaming response for agent={agent.name}, conversation={conversation_id}"
            )
            
            # Build the function dispatch map for manual function-call handling
            function_dispatch = {}
            if self.enable_agentic_retrieval and agentic_function:
                function_dispatch["agentic_retrieval"] = agentic_function

            # Use a while loop to handle multiple sequential function calls
            input_items = []  # Additional input (function call outputs)
            previous_response_id = None

            while True:
                # Build response creation parameters
                response_kwargs = {
                    "conversation": conversation_id,
                    "stream": True,
                    "extra_body": {
                        "agent_reference": {
                            "name": agent.name,
                            "version": agent.version,
                            "type": "agent_reference",
                        }
                    },
                }
                if previous_response_id:
                    response_kwargs["previous_response_id"] = previous_response_id
                if input_items:
                    response_kwargs["input"] = input_items

                response_stream = await openai_client.responses.create(**response_kwargs)

                has_function_calls = False
                function_outputs = []
                current_response_id = None

                async for event in response_stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "response.output_text.delta":
                        yield event.delta

                    elif event_type == "response.output_item.done":
                        item = getattr(event, "item", None)
                        if item and getattr(item, "type", "") == "function_call":
                            has_function_calls = True
                            func_name = item.name
                            func_args_str = getattr(item, "arguments", "{}")
                            call_id = item.call_id
                            logging.info(f"Function call: {func_name}(args={func_args_str[:200]})")

                            # Dispatch function call
                            handler = function_dispatch.get(func_name)
                            if handler:
                                try:
                                    args = json.loads(func_args_str) if func_args_str else {}
                                    result = handler(**args)
                                    function_outputs.append({
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": result if isinstance(result, str) else json.dumps(result),
                                    })
                                except Exception as func_err:
                                    logging.error(f"Function {func_name} failed: {func_err}", exc_info=True)
                                    function_outputs.append({
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": json.dumps({"error": str(func_err)}),
                                    })
                            else:
                                logging.warning(f"No handler for function: {func_name}")
                                function_outputs.append({
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": json.dumps({"error": f"Unknown function: {func_name}"}),
                                })

                    elif event_type == "response.completed":
                        response_obj = getattr(event, "response", None)
                        if response_obj:
                            current_response_id = getattr(response_obj, "id", None)
                        break

                    elif event_type == "response.failed":
                        error_info = getattr(event, "error", None) or event
                        logging.error(f"Response stream failed: {error_info}")
                        raise Exception(f"Agent response failed: {error_info}")

                # If there were function calls, submit outputs and loop for next response
                if has_function_calls and function_outputs:
                    previous_response_id = current_response_id
                    input_items = function_outputs
                    logging.info(f"Submitting {len(function_outputs)} function outputs, continuing...")
                    continue
                else:
                    # No more function calls — done
                    break

            logging.debug("Streaming complete.")

            # Collect final conversation messages for state persistence
            logging.debug("Fetching conversation items for state persistence...")
            conv["messages"] = []
            try:
                items = await openai_client.conversations.items.list(
                    conversation_id=conversation_id
                )
                async for msg in items:
                    role = getattr(msg, "role", None)
                    content_text = ""
                    content_list = getattr(msg, "content", None)
                    if content_list:
                        for content_item in content_list:
                            text_attr = getattr(content_item, "text", None)
                            if text_attr:
                                content_text = text_attr if isinstance(text_attr, str) else getattr(text_attr, "value", str(text_attr))
                                break
                    if not content_text:
                        content_text = getattr(msg, "text", "") or ""
                    if content_text and role:
                        logging.debug(f"Retrieved message: role={role}, text={content_text[:100]!r}")
                        conv["messages"].append({
                            "role": role,
                            "text": content_text,
                        })
            except Exception as exc:
                logging.warning(f"Failed to fetch conversation items: {exc}")

            logging.debug(f"Final conversation messages: {conv['messages']}")

            if self.user_context:
                conv['user_context'] = self.user_context

            if create_agent:
                logging.debug(f"Deleting agent: name={agent.name}, version={agent.version}")
                await project_client.agents.delete_version(
                    agent_name=agent.name, version=agent.version
                )
                logging.debug("Agent deletion complete.")
