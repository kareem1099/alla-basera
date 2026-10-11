from .BaseController import BaseController
from models.dbschemas import Project, DataChunk, RetrievedDocument
from stores.llm.LLMEnums import DocumentTypeEnum
from stores.rerank.RerankEnums import RerankTargetEnums
from helpers.arabic_text import build_keyword_query, normalize_arabic
from utils.metrics import record_rag_answer
from typing import List
import json
import re


class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, embedding_client,
                 template_parser, rerank_client=None, fallback_generation_client=None):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.fallback_generation_client = fallback_generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.rerank_client = rerank_client

    def create_collection_name(self, project_id: int):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()

    async def create_vector_db_collection(self, project: Project, do_reset: bool = False):
        collection_name = self.create_collection_name(project_id=project.project_id)

        return await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(
            collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )

    async def get_indexed_chunk_ids(self, project: Project) -> set:
        if not hasattr(self.vectordb_client, "get_indexed_chunk_ids"):
            return set()
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.get_indexed_chunk_ids(collection_name=collection_name)

    async def create_vector_db_index(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.create_vector_index(collection_name=collection_name)

    async def refresh_keyword_stats(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.refresh_keyword_stats(collection_name=collection_name)

    async def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int]):

        collection_name = self.create_collection_name(project_id=project.project_id)

        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]

        vectors = self.embedding_client.embed_text(
            text=texts,
            document_type=DocumentTypeEnum.DOCUMENT.value,
        )

        if not vectors or len(vectors) == 0:
            return False

        return await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        collection_name = self.create_collection_name(project_id=project.project_id)

        vectors = self.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.QUERY.value,
        )

        if not vectors or len(vectors) == 0:
            return False

        query_vector = None
        if isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]

        if not query_vector:
            return False

        candidates_limit = limit
        if self.rerank_client:
            candidates_limit = max(limit, self.app_settings.RERANK_CANDIDATES)

        results = await self.vectordb_client.hybrid_search(
            collection_name=collection_name,
            vector=query_vector,
            query_text=build_keyword_query(text),
            limit=candidates_limit,
            one_per_parent=self.app_settings.CONTEXT_WINDOW_CHILDREN != 0,
        )

        if not results:
            return False

        if self.rerank_client:
            return await self.rerank_documents(query=text, documents=results, limit=limit)

        return self.select_documents(documents=results, limit=limit)

    def select_documents(self, documents: List[RetrievedDocument], limit: int):

        max_per_parent = (self.app_settings.CONTEXT_MAX_CHILDREN_PER_PARENT
                          if self.app_settings.CONTEXT_WINDOW_CHILDREN == 0 else None)

        selected, parent_counts = [], {}
        for document in documents:
            parent_id = (document.metadata or {}).get("parent_id")
            if max_per_parent is not None and parent_counts.get(parent_id, 0) >= max_per_parent:
                continue
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
            selected.append(document)
            if len(selected) == limit:
                break

        return selected

    async def rerank_documents(self, query: str, documents: List[RetrievedDocument], limit: int):

        if self.app_settings.RERANK_TARGET == RerankTargetEnums.CHUNK.value:
            texts = [(doc.metadata or {}).get("matched_chunk_text", doc.text) for doc in documents]
        else:
            texts = [doc.text for doc in documents]

        ranked = await self.rerank_client.rerank(query=query, documents=texts, top_n=len(documents))

        if not ranked:
            return self.select_documents(documents=documents, limit=limit)

        reranked = []
        for index, score in ranked:
            document = documents[index]
            document.metadata = {
                **(document.metadata or {}),
                "hybrid_rank": index + 1,
                "hybrid_score": document.score,
            }
            document.score = score
            reranked.append(document)

        return self.select_documents(documents=reranked, limit=limit)

    async def answer_rag_question(self, project: Project, query: str, limit: int = 10):

        answer, full_prompt, chat_history = None, None, None

        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history, None

        system_prompt = self.template_parser.get("rag", "system_prompt")

        if getattr(self.generation_client, "supports_grounded_generation", False):
            return self.answer_grounded(query=query, documents=retrieved_documents,
                                        system_prompt=system_prompt)

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "title": (doc.metadata or {}).get("title", ""),
                "chunk_text": doc.text,
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query,
        })

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history,
        )

        return answer, full_prompt, chat_history, None

    def answer_grounded(self, query: str, documents: List[RetrievedDocument], system_prompt: str):

        grounded_documents = [
            {
                "id": str((doc.metadata or {}).get(
                    "matched_chunk_id" if self.app_settings.CONTEXT_WINDOW_CHILDREN == 0 else "parent_id",
                    idx + 1)),
                "title": (doc.metadata or {}).get("title", ""),
                "text": self.build_context_window(doc),
            }
            for idx, doc in enumerate(documents)
        ]

        result, answering_client = None, None
        for client in (self.generation_client, self.fallback_generation_client):
            if client is None:
                continue
            result = client.generate_grounded(
                query=query,
                documents=grounded_documents,
                system_prompt=system_prompt,
                max_output_tokens=self.app_settings.GROUNDED_MAX_OUTPUT_TOKENS,
                temperature=self.app_settings.GROUNDED_TEMPERATURE,
                frequency_penalty=self.app_settings.GROUNDED_FREQUENCY_PENALTY,
                presence_penalty=self.app_settings.GROUNDED_PRESENCE_PENALTY,
            )
            if result:
                answering_client = client
                break

        if not result:
            return None, None, None, None

        answer, citations = self.remove_special_tokens(
            answer=result["text"], citations=result.get("citations") or [])
        answer, citations, loop_detected = self.cut_repetition_loop(
            answer=answer, citations=citations)
        quotes_checked, unverified_quotes = self.verify_quotes(
            answer=answer, documents_texts=[doc["text"] for doc in grounded_documents])
        attributions_checked, unverified_attributions = self.verify_attributions(
            answer=answer, documents_texts=[doc["text"] for doc in grounded_documents])
        finish_reason = result.get("finish_reason")
        citation_markup_complete = result.get("citation_markup_complete", True)
        supports_citations = getattr(answering_client, "supports_citations", True)
        grounding_ratio = (self.compute_grounding_ratio(answer=answer, citations=citations)
                           if supports_citations else None)
        refusal_message = self.template_parser.get("rag", "refusal_message")

        complete = finish_reason != "MAX_TOKENS" and citation_markup_complete and not loop_detected

        if (supports_citations
                and answer.strip() != refusal_message.strip()
                and grounding_ratio < self.app_settings.GROUNDING_MIN_RATIO):
            answer = refusal_message
            citations = []
            grounding_ratio = 0.0
            complete = True

        grounding = {
            "model": result.get("model") or answering_client.generation_model_id,
            "citations": citations,
            "grounding_ratio": grounding_ratio,
            "complete": complete,
            "loop_detected": loop_detected,
            "quotes_checked": quotes_checked,
            "unverified_quotes": unverified_quotes,
            "attributions_checked": attributions_checked,
            "unverified_attributions": unverified_attributions,
            "finish_reason": finish_reason,
            "citation_markup_complete": citation_markup_complete,
            "documents": [{"id": doc["id"], "title": doc["title"]} for doc in grounded_documents],
        }

        if answer.strip() == refusal_message.strip():
            outcome = "refused"
        elif not complete:
            outcome = "incomplete"
        else:
            outcome = "answered"
        record_rag_answer(
            model=grounding["model"],
            outcome=outcome,
            unverified_quotes=len(unverified_quotes),
            unverified_attributions=len(unverified_attributions),
        )

        return answer, query, system_prompt, grounding

    def build_context_window(self, doc: RetrievedDocument):

        window = self.app_settings.CONTEXT_WINDOW_CHILDREN
        metadata = doc.metadata or {}
        children_ids = metadata.get("children_ids") or []
        children_texts = metadata.get("children_texts") or []

        if window is None or metadata.get("matched_chunk_id") not in children_ids:
            return doc.text

        index = children_ids.index(metadata["matched_chunk_id"])
        selected = children_texts[max(0, index - window):index + window + 1]

        start = doc.text.find(selected[0])
        end = doc.text.find(selected[-1], max(start, 0))
        if start == -1 or end == -1:
            return "\n".join(selected)

        return doc.text[start:end + len(selected[-1])]

    @staticmethod
    def remove_special_tokens(answer: str, citations: list):

        matches = list(re.finditer(r'<\|?[A-Z][A-Z_]*_TOKEN\|?>', answer))
        if not matches:
            return answer, citations

        def shift(position: int) -> int:
            return position - sum(m.end() - m.start() for m in matches if m.end() <= position)

        cleaned = re.sub(r'<\|?[A-Z][A-Z_]*_TOKEN\|?>', '', answer)
        shifted = [
            {**citation, "start": shift(citation["start"]), "end": shift(citation["end"])}
            for citation in citations
        ]
        return cleaned, shifted

    @staticmethod
    def verify_quotes(answer: str, documents_texts: list, min_length: int = 12):

        def normalize_quote(text: str, remove_brackets: bool = True) -> str:
            if remove_brackets:
                text = re.sub(r'\[[^\]]*\]|\([^)]*\)', ' ', text)
            text = re.sub(r'[0-9٠-٩]+', ' ', text)
            return normalize_arabic(text)

        quotes = []
        for pattern in (r'﴿([^﴾]+)﴾', r'«([^»]+)»', r'"([^"]+)"', r'“([^”]+)”'):
            quotes.extend(re.findall(pattern, answer))
        quotes.extend(line.lstrip('> ').strip() for line in answer.split('\n') if line.startswith('>'))

        corpus = ' | '.join(
            normalize_quote(text, remove_brackets=remove)
            for text in documents_texts for remove in (True, False)
        )

        checked, unverified, seen = 0, [], set()
        for quote in quotes:
            fragments = [
                normalize_quote(fragment)
                for fragment in re.split(r'\.\.\.|…|\[\s*\.\.\.\s*\]', quote)
            ]
            fragments = [f for f in fragments if len(f) >= min_length]
            key = ' | '.join(fragments)
            if not fragments or key in seen:
                continue
            seen.add(key)
            checked += 1
            if not all(f in corpus for f in fragments):
                unverified.append(quote.strip()[:300])

        return checked, unverified

    @staticmethod
    def verify_attributions(answer: str, documents_texts: list, min_length: int = 12,
                            fragment_length: int = 25, window: int = 600):

        attribution_pattern = re.compile(r'\(\s*(متفق عليه|(?:أخرجه|أخرجهما|اخرجه|رواه)[^)]*)\)')
        verbs = {'اخرجه', 'اخرجهما', 'رواه'}
        sources = []

        def mark(text: str) -> str:
            def replace(match):
                key = frozenset(w for w in normalize_arabic(match.group(1)).split() if w not in verbs)
                sources.append((key, match.group(1).strip()))
                return f' xsrcx{len(sources) - 1}x '
            return normalize_arabic(attribution_pattern.sub(replace, text))

        corpus = ' '.join(mark(text) for text in documents_texts)
        marked_answer = mark(answer)

        checked, unverified = 0, []
        for match in re.finditer(r'xsrcx(\d+)x', marked_answer):
            before = re.sub(r'xsrcx\d+x', ' ', marked_answer[:match.start()])
            fragment = re.sub(r'\s+', ' ', before).strip()[-fragment_length:].strip()
            if len(fragment) < min_length:
                continue

            positions = [m.start() for m in re.finditer(re.escape(fragment), corpus)]
            if not positions:
                continue

            checked += 1
            answer_key, answer_source = sources[int(match.group(1))]
            matched = False
            for position in positions:
                next_source = re.search(r'xsrcx(\d+)x', corpus[position:position + window])
                if next_source and sources[int(next_source.group(1))][0] == answer_key:
                    matched = True
                    break

            if not matched:
                unverified.append(f"...{fragment} ({answer_source})")

        return checked, unverified

    @staticmethod
    def cut_repetition_loop(answer: str, citations: list, min_length: int = 20, max_repeats: int = 3):

        segments = [
            (m.start(), re.sub(r'\s+', ' ', m.group()).strip())
            for m in re.finditer(r'[^.،,؛;\n]+', answer)
        ]

        counts = {}
        first_repeat_at = None
        loop_detected = False
        for start, segment in segments:
            if len(segment) < min_length:
                continue
            counts[segment] = counts.get(segment, 0) + 1
            if counts[segment] == 2 and first_repeat_at is None:
                first_repeat_at = start
            if counts[segment] >= max_repeats:
                loop_detected = True
                break

        if not loop_detected:
            return answer, citations, False

        cut_answer = answer[:first_repeat_at].rstrip(' ،,؛;\n')
        kept_citations = [citation for citation in citations if citation["end"] <= len(cut_answer)]
        return cut_answer, kept_citations, True

    @staticmethod
    def compute_grounding_ratio(answer: str, citations: list):

        answer_length = len(answer)
        if answer_length == 0:
            return 0.0

        spans = []
        for citation in citations:
            start = max(0, citation["start"])
            end = min(answer_length, citation["end"])
            if start < end:
                spans.append((start, end))
        spans.sort()

        if not spans:
            return 0.0

        covered = 0
        current_start, current_end = None, None
        for start, end in spans:
            if current_end is None or start > current_end:
                if current_end is not None:
                    covered += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        covered += current_end - current_start

        return round(min(covered / answer_length, 1.0), 3)
