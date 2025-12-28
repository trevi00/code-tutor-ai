"""RAG Engine for Code Tutor AI - Retrieval-Augmented Generation"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

from code_tutor.ml.rag.vector_store import FAISSVectorStore
from code_tutor.ml.rag.pattern_knowledge import PatternKnowledgeBase
from code_tutor.ml.embeddings import TextEmbedder, CodeEmbedder
from code_tutor.ml.config import get_ml_config

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine for algorithm pattern retrieval and augmented generation.

    Features:
    - FAISS vector store for fast similarity search
    - Pattern knowledge base with 25 algorithm patterns
    - LangChain integration for LLM generation
    - Korean LLM support (EEVE-Korean or OpenAI fallback)
    """

    def __init__(
        self,
        config=None,
        text_embedder: Optional[TextEmbedder] = None,
        code_embedder: Optional[CodeEmbedder] = None
    ):
        self.config = config or get_ml_config()

        # Initialize embedders
        self._text_embedder = text_embedder
        self._code_embedder = code_embedder

        # Initialize vector store
        self._vector_store = None

        # Initialize pattern knowledge base
        self._pattern_kb = PatternKnowledgeBase(
            data_path=self.config.PATTERN_DATA_PATH
        )

        # LLM will be lazily loaded
        self._llm = None
        self._is_initialized = False

    @property
    def text_embedder(self) -> TextEmbedder:
        """Get text embedder, creating if needed"""
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder(
                model_name=self.config.EMBEDDING_MODEL,
                cache_dir=str(self.config.MODEL_CACHE_DIR)
            )
        return self._text_embedder

    @property
    def code_embedder(self) -> CodeEmbedder:
        """Get code embedder, creating if needed"""
        if self._code_embedder is None:
            self._code_embedder = CodeEmbedder(
                model_name=self.config.CODE_EMBEDDING_MODEL,
                cache_dir=str(self.config.MODEL_CACHE_DIR)
            )
        return self._code_embedder

    @property
    def vector_store(self) -> FAISSVectorStore:
        """Get vector store, creating if needed"""
        if self._vector_store is None:
            self._vector_store = FAISSVectorStore(
                dimension=self.config.EMBEDDING_DIMENSION,
                metric="cosine",
                index_path=self.config.FAISS_INDEX_PATH
            )
        return self._vector_store

    @property
    def pattern_kb(self) -> PatternKnowledgeBase:
        """Get pattern knowledge base"""
        return self._pattern_kb

    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the RAG engine with pattern embeddings.

        Args:
            force_rebuild: Whether to rebuild embeddings even if cached
        """
        if self._is_initialized and not force_rebuild:
            return

        logger.info("Initializing RAG engine...")

        # Check if index exists
        index_path = self.config.FAISS_INDEX_PATH
        if index_path.exists() and not force_rebuild:
            try:
                self.vector_store.load()
                self._is_initialized = True
                logger.info("Loaded existing vector store")
                return
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, rebuilding...")

        # Build pattern embeddings
        self._build_pattern_index()
        self._is_initialized = True

    def _build_pattern_index(self):
        """Build FAISS index from pattern knowledge base"""
        logger.info("Building pattern index...")

        # Get documents from pattern KB
        documents = self._pattern_kb.to_documents()

        # Create embeddings
        contents = [doc["content"] for doc in documents]
        embeddings = self.text_embedder.embed_batch(contents)

        # Add to vector store
        ids = [doc["id"] for doc in documents]
        metadata = [doc["metadata"] for doc in documents]

        self.vector_store.add(embeddings, ids, metadata)

        # Save index
        self.config.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save()

        logger.info(f"Built index with {len(documents)} patterns")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> List[Dict]:
        """
        Retrieve relevant patterns for a query.

        Args:
            query: Natural language query
            top_k: Number of results (default from config)
            threshold: Minimum similarity threshold (default from config)

        Returns:
            List of relevant patterns with scores
        """
        self.initialize()

        top_k = top_k or self.config.RAG_TOP_K
        threshold = threshold or self.config.RAG_SIMILARITY_THRESHOLD

        # Embed query
        query_emb = self.text_embedder.embed(query)

        # Search vector store
        results = self.vector_store.search(query_emb, top_k, threshold)

        # Enrich with full pattern data
        enriched_results = []
        for result in results:
            pattern = self._pattern_kb.get_pattern(result["id"])
            if pattern:
                enriched_results.append({
                    **pattern,
                    "score": result["score"]
                })

        return enriched_results

    def retrieve_by_code(
        self,
        code: str,
        language: str = "python",
        top_k: int = None,
        threshold: float = None
    ) -> List[Dict]:
        """
        Retrieve relevant patterns based on code similarity.

        Args:
            code: Source code snippet
            language: Programming language
            top_k: Number of results
            threshold: Minimum similarity threshold

        Returns:
            List of relevant patterns with scores
        """
        # Build code embeddings if needed
        if self._pattern_kb._embeddings is None:
            self._pattern_kb.build_embeddings(
                self.text_embedder,
                self.code_embedder
            )

        return self._pattern_kb.find_similar_by_code(
            code=code,
            language=language,
            top_k=top_k or self.config.RAG_TOP_K,
            threshold=threshold or self.config.RAG_SIMILARITY_THRESHOLD
        )

    def _get_llm(self):
        """Get or create LLM instance"""
        if self._llm is not None:
            return self._llm

        if self.config.USE_LOCAL_LLM:
            self._llm = self._create_local_llm()
        else:
            self._llm = self._create_openai_llm()

        return self._llm

    def _create_local_llm(self):
        """Create local LLM (EEVE-Korean)"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            logger.info(f"Loading local LLM: {self.config.LOCAL_LLM_MODEL}")

            # Quantization config for 4-bit
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.LOCAL_LLM_MODEL,
                cache_dir=str(self.config.MODEL_CACHE_DIR)
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.config.LOCAL_LLM_MODEL,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=str(self.config.MODEL_CACHE_DIR)
            )

            return {"model": model, "tokenizer": tokenizer, "type": "local"}

        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            logger.info("Falling back to OpenAI")
            return self._create_openai_llm()

    def _create_openai_llm(self):
        """Create OpenAI LLM"""
        try:
            from openai import OpenAI

            if not self.config.OPENAI_API_KEY:
                logger.warning("OpenAI API key not configured")
                return None

            client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            return {"client": client, "type": "openai"}

        except ImportError:
            logger.error("OpenAI package not installed")
            return None

    def generate(
        self,
        query: str,
        context: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> str:
        """
        Generate response using RAG.

        Args:
            query: User query
            context: Retrieved patterns (will retrieve if None)
            system_prompt: Custom system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response string
        """
        # Retrieve context if not provided
        if context is None:
            context = self.retrieve(query)

        # Build context string
        context_str = self._build_context_string(context)

        # Default system prompt
        if system_prompt is None:
            system_prompt = """당신은 알고리즘 튜터입니다. 학생의 질문에 대해 주어진 패턴 정보를 활용하여
친절하고 교육적인 답변을 제공하세요. 코드 예제를 포함하고, 한국어로 답변하세요.

관련 패턴 정보:
{context}"""

        prompt = system_prompt.format(context=context_str)

        llm = self._get_llm()
        if llm is None:
            return self._generate_fallback(query, context)

        if llm["type"] == "openai":
            return self._generate_openai(llm["client"], prompt, query, max_tokens)
        else:
            return self._generate_local(llm, prompt, query, max_tokens)

    def _build_context_string(self, patterns: List[Dict]) -> str:
        """Build context string from patterns"""
        if not patterns:
            return "관련 패턴이 없습니다."

        parts = []
        for i, pattern in enumerate(patterns, 1):
            parts.append(f"""
### {i}. {pattern['name']} ({pattern['name_ko']})
- 설명: {pattern['description_ko']}
- 시간 복잡도: {pattern['time_complexity']}
- 공간 복잡도: {pattern['space_complexity']}
- 활용 사례: {', '.join(pattern['use_cases'])}

```python
{pattern['example_code']}
```
""")
        return "\n".join(parts)

    def _generate_openai(self, client, system_prompt: str, query: str, max_tokens: int) -> str:
        """Generate using OpenAI API"""
        try:
            response = client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._generate_fallback(query, [])

    def _generate_local(self, llm: Dict, system_prompt: str, query: str, max_tokens: int) -> str:
        """Generate using local LLM"""
        try:
            model = llm["model"]
            tokenizer = llm["tokenizer"]

            # Format prompt for EEVE
            full_prompt = f"""### System:
{system_prompt}

### User:
{query}

### Assistant:
"""
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "### Assistant:" in response:
                response = response.split("### Assistant:")[-1].strip()

            return response

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            return self._generate_fallback(query, [])

    def _generate_fallback(self, query: str, context: List[Dict]) -> str:
        """Fallback response when LLM is unavailable"""
        if not context:
            return f"죄송합니다. '{query}'에 대한 관련 패턴을 찾지 못했습니다."

        # Simple pattern-based response
        pattern = context[0]
        return f"""'{query}'에 관련된 알고리즘 패턴을 찾았습니다:

## {pattern['name']} ({pattern['name_ko']})

**설명**: {pattern['description_ko']}

**시간 복잡도**: {pattern['time_complexity']}
**공간 복잡도**: {pattern['space_complexity']}

**활용 사례**:
{chr(10).join('- ' + uc for uc in pattern['use_cases'])}

**예제 코드**:
```python
{pattern['example_code']}
```

**키워드**: {', '.join(pattern['keywords'])}
"""

    def analyze_code(
        self,
        code: str,
        language: str = "python",
        analysis_type: str = "pattern"
    ) -> Dict[str, Any]:
        """
        Analyze code and identify patterns.

        Args:
            code: Source code to analyze
            language: Programming language
            analysis_type: Type of analysis ("pattern", "complexity", "full")

        Returns:
            Analysis results with identified patterns and suggestions
        """
        # Retrieve similar patterns
        patterns = self.retrieve_by_code(code, language, top_k=5)

        result = {
            "detected_patterns": [],
            "suggestions": [],
            "complexity_estimate": None
        }

        for pattern in patterns:
            if pattern.get("similarity", 0) > 0.7:
                result["detected_patterns"].append({
                    "pattern": pattern["name"],
                    "pattern_ko": pattern["name_ko"],
                    "confidence": pattern["similarity"],
                    "description": pattern["description_ko"]
                })

        # Generate suggestions based on patterns
        if result["detected_patterns"]:
            main_pattern = result["detected_patterns"][0]
            full_pattern = self._pattern_kb.get_pattern(
                patterns[0]["id"]
            )
            if full_pattern:
                result["complexity_estimate"] = {
                    "time": full_pattern["time_complexity"],
                    "space": full_pattern["space_complexity"]
                }

                # Suggest related patterns
                result["suggestions"].append(
                    f"이 코드는 '{main_pattern['pattern_ko']}' 패턴을 사용하고 있습니다."
                )

        return result

    def get_pattern_explanation(
        self,
        pattern_id: str,
        detail_level: str = "medium"
    ) -> Dict[str, Any]:
        """
        Get detailed explanation of a pattern.

        Args:
            pattern_id: Pattern ID
            detail_level: "brief", "medium", or "detailed"

        Returns:
            Pattern explanation with examples
        """
        pattern = self._pattern_kb.get_pattern(pattern_id)
        if not pattern:
            return {"error": f"Pattern not found: {pattern_id}"}

        explanation = {
            "name": pattern["name"],
            "name_ko": pattern["name_ko"],
            "description": pattern["description_ko"],
            "use_cases": pattern["use_cases"],
            "complexity": {
                "time": pattern["time_complexity"],
                "space": pattern["space_complexity"]
            }
        }

        if detail_level in ["medium", "detailed"]:
            explanation["example_code"] = pattern["example_code"]
            explanation["keywords"] = pattern["keywords"]

        if detail_level == "detailed":
            # Generate additional explanation using LLM
            query = f"{pattern['name_ko']} 패턴에 대해 상세히 설명해주세요."
            explanation["detailed_explanation"] = self.generate(
                query,
                context=[pattern],
                max_tokens=1024
            )

        return explanation

    def unload(self):
        """Unload all models to free memory"""
        if self._text_embedder:
            self._text_embedder.unload()
            self._text_embedder = None

        if self._code_embedder:
            self._code_embedder.unload()
            self._code_embedder = None

        if self._llm and self._llm.get("type") == "local":
            del self._llm["model"]
            del self._llm["tokenizer"]

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._llm = None
        logger.info("RAG engine unloaded")
