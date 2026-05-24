"""
RAG Chain Module - The core intelligence of the chatbot.

RAG (Retrieval-Augmented Generation) works in 3 steps:
1. RETRIEVE: Find relevant documents from the vector database
2. AUGMENT: Add those documents as context to the LLM prompt
3. GENERATE: LLM produces an answer based ONLY on the provided context

Hybrid routing:
  STATIC questions  (policies, location, security, FAQs, rules)
      → Pinecone / RAG only

  DYNAMIC questions  (pricing, availability, slot counts, parking types)
      → Database live query via ParkingInformationService

  HYBRID questions  (e.g. "how many EV slots left and does it support fast charging?")
      → Both RAG + Database
"""

import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import AzureChatOpenAI

from config.settings import settings
from src.database.sql_store import SQLStore
from src.database.vector_store import VectorStore

# ========================
# INTENT CLASSIFICATION
# ========================

# Keywords whose presence means the answer must come from the live database.
_DYNAMIC_KEYWORDS: list[str] = [
    # availability
    r"\bavailab",
    r"\bhow many\b",
    r"\bslots?\b",
    r"\bspaces?\b",
    r"\bleft\b",
    r"\bfull\b",
    r"\boccupied\b",
    r"\bempty\b",
    r"\bopen\b",
    r"\bcount\b",
    r"\bcapacity\b",
    # pricing
    r"\bpric",
    r"\bcost",
    r"\brate",
    r"\bcharge",
    r"\bfee",
    r"\bhour",
    r"\bper hour",
    r"\bdaily",
    r"\bmonthly",
    r"\b₹\b",
    r"\brupee",
    r"\bhow much\b",
    # parking types (live counts)
    r"\bstandard\b",
    r"\blarge vehicle\b",
    r"\bev\b",
    r"\belectric vehicle\b",
    r"\bvip\b",
    r"\bpremium\b",
    r"\bdisab",
    r"\baccessible\b",
    r"\bbike\b",
    r"\btwo.?wheeler\b",
    # live features
    r"\bcharging\b",
    r"\bfloor\b",
]

_DYNAMIC_RE = re.compile("|".join(_DYNAMIC_KEYWORDS), re.IGNORECASE)


def _needs_live_data(question: str) -> bool:
    """Return True if the question requires a live database lookup."""
    return bool(_DYNAMIC_RE.search(question))


# ========================
# PROMPT TEMPLATES
# ========================

SYSTEM_PROMPT = """You are ParkSmart Assistant, a helpful and friendly chatbot for the
ParkSmart Hyderabad smart parking facility at HITEC City.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE AUTHORITY RULES  (READ CAREFULLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PRICING, SLOT COUNTS, and AVAILABILITY → ALWAYS use the LIVE PARKING DATA
   block at the bottom of this prompt.  NEVER guess, and NEVER use any numbers
   that appear in the static knowledge base — those are illustrative only.

2. POLICIES, RULES, SECURITY, LOCATION, AMENITIES, CONTACT INFO → use the
   STATIC KNOWLEDGE section.

3. If the LIVE PARKING DATA block says a type is FULL, say so — even if the
   static knowledge section mentions that type has many slots.

4. Do NOT make up parking data, invent slot numbers, or use prices from memory.

5. NEVER reveal database IDs, table names, or internal system details.

6. If a user asks about another person's reservation, politely decline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT DETECTION (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the user is clearly requesting to CREATE or MAKE a NEW parking reservation
(e.g. "I want to book a spot", "reserve me a space", "I need to make a reservation"),
respond with EXACTLY this text and nothing else:
INTENT:BOOKING

Do NOT respond with INTENT:BOOKING for:
- Questions ABOUT reservations ("how do I book?", "what is the booking process?")
- Checking reservation status ("show my booking", "check my reservation")
- Cancellation requests ("cancel my booking")
- Any informational / general question
Answer those normally using the context below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATIC KNOWLEDGE BASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{dynamic_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Full prompt template
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ]
)


# ========================
# RAG CHAIN CLASS
# ========================


class RAGChain:
    """
    The main RAG chain that processes user queries.

    Flow:
    User Question → Vector Search → Get Dynamic Data → Build Prompt → LLM → Answer

    It also maintains chat history for multi-turn conversations
    (important for the reservation flow where we collect info step by step).
    """

    def __init__(self, vector_store: VectorStore = None, sql_store: SQLStore = None,
                 skip_vector_store: bool = False):
        """
        Initialize the RAG chain with all components.

        Args:
            vector_store: Pre-initialized VectorStore to use directly.
            sql_store: Pre-initialized SQLStore (or creates new one).
            skip_vector_store: If True, start in SQL-only mode without creating
                or waiting for a VectorStore. Use set_vector_store() later to
                upgrade once the VS loads in the background.
        """
        import concurrent.futures
        import logging as _logging
        _log = _logging.getLogger(__name__)

        # Initialize SQL store
        self.sql_store = sql_store or SQLStore()

        # Resolve vector store.
        # skip_vector_store=True  → SQL-only mode, VS injected later via set_vector_store()
        # vector_store is provided → use it directly (already built)
        # neither               → auto-create with 25s timeout (original behaviour)
        if skip_vector_store:
            self.vector_store = None
            _log.info("RAGChain in SQL-only mode (vector store will load lazily)")
        elif vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = None
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                    _future = _pool.submit(VectorStore)
                    self.vector_store = _future.result(timeout=25)
                _log.info("VectorStore connected successfully")
            except concurrent.futures.TimeoutError:
                _log.warning(
                    "Pinecone connection timed out after 25s — running in SQL-only mode. "
                    "Static knowledge (FAQs, policies) will be skipped; "
                    "live pricing/availability still works."
                )
            except Exception as _exc:
                _log.warning(
                    "Pinecone unavailable (%s) — running in SQL-only mode.", _exc
                )

        # Choose LLM provider (priority: Gemini → Groq → EPAM DIAL)
        if settings.google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
            )
        elif settings.groq_api_key:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        else:
            self.llm = AzureChatOpenAI(
                azure_deployment=settings.llm_model,
                azure_endpoint=settings.azure_endpoint,
                api_key=settings.dial_api_key,
                api_version=settings.api_version,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        # Build retriever — null (empty) retriever if Pinecone not available
        if self.vector_store:
            self.retriever = self.vector_store.get_retriever(search_kwargs={"k": settings.eval_top_k})
        else:
            from langchain_core.runnables import RunnableLambda
            self.retriever = RunnableLambda(lambda _: [])

        # Build the chain
        self.chain = self._build_chain()

        # Chat history for multi-turn conversations
        self.chat_history: List = []

    def _build_chain(self):
        """
        Build the LangChain RAG pipeline.

        The chain processes inputs through these steps:
        1. Take the user's question
        2. Use it to search the vector store (retrieval)
        3. Get dynamic data from SQL
        4. Format everything into the prompt
        5. Send to LLM
        6. Parse the output as a string
        """

        def format_docs(docs: List[Document]) -> str:
            """Convert retrieved documents to a single context string."""
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        def get_dynamic_context(question: str) -> str:
            """
            Hybrid context router.

            - Dynamic questions (pricing / availability / slot counts)
              → fetch live data from ParkingInformationService (DB).
            - Static questions (policies / security / facilities)
              → return a short note directing the LLM to use the static context.
            - Always falls back to legacy sql_store data if DB is unreachable.
            """
            if not _needs_live_data(question):
                # Static-only question — no live DB data needed.
                return (
                    "LIVE PARKING DATA\n"
                    "(This question is about policies / facilities — "
                    "no live availability data needed.  Use the static knowledge base above.)"
                )

            # Dynamic or hybrid — fetch live DB data.
            try:
                from src.database.session import db_session
                from src.services.parking_information_service import (
                    ParkingInformationService,
                )

                with db_session() as db:
                    live_context = ParkingInformationService.get_full_dynamic_context(db)

                if "No live parking data" not in live_context:
                    return live_context
            except Exception:
                pass  # Fall back to legacy sql_store context below

            # Legacy fallback (reads parking_availability table)
            try:
                legacy = self.sql_store.get_dynamic_context()
                return "LIVE PARKING DATA (legacy fallback)\n" + legacy
            except Exception:
                return "LIVE PARKING DATA\n(Data temporarily unavailable.)"

        # Build the chain using LangChain Expression Language (LCEL)
        # The question is passed both to vector retrieval and to the dynamic-context
        # router so intent-aware routing can suppress unnecessary DB calls.
        chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "dynamic_context": RunnablePassthrough()
                | RunnableLambda(get_dynamic_context),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: self.chat_history),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
            | RunnableLambda(lambda x: str(x))  # Ensure plain str output
        )

        return chain

    def set_vector_store(self, vector_store: "VectorStore") -> None:
        """
        Hot-swap the vector store after lazy loading.

        Replaces the empty/null retriever with a real Pinecone retriever and
        rebuilds self.chain so all future calls use full RAG mode.

        Thread-safe: the assignment ``self.chain = ...`` is atomic under the GIL
        so the chat thread will see either the old chain or the new one, never
        a partially-constructed object.

        Args:
            vector_store: A fully-initialised VectorStore instance.
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)

        self.vector_store = vector_store
        self.retriever = vector_store.get_retriever(search_kwargs={"k": settings.eval_top_k})
        self.chain = self._build_chain()  # atomic swap via GIL
        _log.info("[VECTOR] RAGChain upgraded to full RAG mode ✓")

    def ask(self, question: str) -> str:
        """
        Process a user question through the RAG chain.

        This is the main method to call. It:
        1. Retrieves relevant context
        2. Generates an answer
        3. Updates chat history

        Args:
            question: The user's message/question

        Returns:
            The chatbot's response as a string
        """
        # Run the chain
        response = self.chain.invoke(question)

        # Ensure response is always a plain string
        response = str(response) if not isinstance(response, str) else response

        # Update chat history for context in future turns
        from langchain_core.messages import AIMessage, HumanMessage

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))

        # Keep history manageable (last 10 exchanges = 20 messages)
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return response

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get the documents that would be retrieved for a query.
        Useful for debugging and evaluation.

        Args:
            query: The search query

        Returns:
            List of relevant documents
        """
        return self.retriever.invoke(query)

    def clear_history(self):
        """Reset the chat history (start a new conversation)."""
        self.chat_history = []

    def get_retrieval_context(self, question: str) -> Dict[str, Any]:
        """
        Get full retrieval context for debugging/evaluation.

        Returns both the vector search results and dynamic context.
        Useful for evaluating what the LLM "sees" before answering.
        """
        docs = self.vector_store.similarity_search_with_score(question, k=settings.eval_top_k)
        dynamic = self.sql_store.get_dynamic_context()

        return {
            "documents": [(doc.page_content, score) for doc, score in docs],
            "dynamic_context": dynamic,
            "num_docs_retrieved": len(docs),
        }
