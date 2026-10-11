from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import (
    DistanceMethodEnums,
    PgVectorTableSchemeEnums,
    PgVectorDistanceMethodEnums,
    PgVectorIndexTypeEnums,
    PgVectorHybridSearchEnums,
)
from models.dbschemas import RetrievedDocument
from helpers.arabic_text import normalize_arabic_sql
from sqlalchemy.sql import text as sql_text
from sqlalchemy.orm import sessionmaker
import logging
import json
from typing import List


class PGVectorProvider(VectorDBInterface):

    def __init__(self, db_client: sessionmaker, distance_method: str,
                 default_vector_size: int = 786, index_threshold: int = 100):

        self.db_client = db_client
        self.default_vector_size = default_vector_size
        self.index_threshold = index_threshold

        self.distance_method = None
        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = PgVectorDistanceMethodEnums.COSINE.value
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = PgVectorDistanceMethodEnums.DOT.value
        else:
            raise ValueError(f"Unsupported distance method: {distance_method}")

        self.logger = logging.getLogger(__name__)

        self.pgvector_table_prefix = PgVectorTableSchemeEnums._PREFIX.value
        self.ts_config = PgVectorHybridSearchEnums.TS_CONFIG.value

    def default_index_name(self, collection_name: str) -> str:
        return f"{collection_name}_{PgVectorTableSchemeEnums.VECTOR.value}_idx"

    def default_tsv_index_name(self, collection_name: str) -> str:
        return f"{collection_name}_{PgVectorTableSchemeEnums.TEXT_TSV.value}_idx"

    def default_bm25_terms_table(self, collection_name: str) -> str:
        return f"{collection_name}_{PgVectorHybridSearchEnums.BM25_TERMS_SUFFIX.value}"

    def default_bm25_stats_table(self, collection_name: str) -> str:
        return f"{collection_name}_{PgVectorHybridSearchEnums.BM25_STATS_SUFFIX.value}"

    async def connect(self):
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(
                    "CREATE EXTENSION IF NOT EXISTS vector"
                ))
                await session.commit()

    async def disconnect(self):
        pass

    async def is_collection_existed(self, collection_name: str) -> bool:
        async with self.db_client() as session:
            check_sql = sql_text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = :table_name)"
            )
            result = await session.execute(check_sql, {"table_name": collection_name})
            return bool(result.scalar_one())

    async def list_all_collections(self) -> List:
        async with self.db_client() as session:
            list_sql = sql_text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE :prefix"
            )
            result = await session.execute(list_sql, {"prefix": f"{self.pgvector_table_prefix}%"})
            return [row[0] for row in result.fetchall()]

    async def get_collection_info(self, collection_name: str) -> dict:
        async with self.db_client() as session:
            count_sql = sql_text(f"SELECT COUNT(*) FROM {collection_name}")
            result = await session.execute(count_sql)
            records_count = result.scalar_one()

            index_sql = sql_text(
                "SELECT indexname FROM pg_indexes WHERE tablename = :table_name"
            )
            index_result = await session.execute(index_sql, {"table_name": collection_name})
            indexes = [row[0] for row in index_result.fetchall()]

            return {
                "table_name": collection_name,
                "records_count": records_count,
                "indexes": indexes,
            }

    async def get_indexed_chunk_ids(self, collection_name: str) -> set:
        if not await self.is_collection_existed(collection_name=collection_name):
            return set()
        async with self.db_client() as session:
            result = await session.execute(sql_text(
                f"SELECT DISTINCT {PgVectorTableSchemeEnums.CHUNK_ID.value} FROM {collection_name}"))
            return {row[0] for row in result.fetchall()}

    async def delete_collection(self, collection_name: str):
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(f"DROP TABLE IF EXISTS {self.default_bm25_terms_table(collection_name)}"))
                await session.execute(sql_text(f"DROP TABLE IF EXISTS {self.default_bm25_stats_table(collection_name)}"))
                await session.execute(sql_text(f"DROP TABLE IF EXISTS {collection_name}"))
                await session.commit()
        return True

    async def is_keyword_stats_existed(self, collection_name: str) -> bool:
        async with self.db_client() as session:
            result = await session.execute(
                sql_text("SELECT to_regclass(:terms) IS NOT NULL AND to_regclass(:stats) IS NOT NULL"),
                {"terms": self.default_bm25_terms_table(collection_name),
                 "stats": self.default_bm25_stats_table(collection_name)},
            )
            return bool(result.scalar_one())

    async def refresh_keyword_stats(self, collection_name: str):

        if not await self.is_collection_existed(collection_name=collection_name):
            return False

        terms_table = self.default_bm25_terms_table(collection_name)
        stats_table = self.default_bm25_stats_table(collection_name)
        tsv_column = PgVectorTableSchemeEnums.TEXT_TSV.value

        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(f"DROP TABLE IF EXISTS {terms_table}"))
                await session.execute(sql_text(f"DROP TABLE IF EXISTS {stats_table}"))
                await session.execute(sql_text(
                    f"CREATE TABLE {terms_table} AS "
                    f"SELECT word AS lexeme, ndoc AS df "
                    f"FROM ts_stat('SELECT {tsv_column} FROM {collection_name}')"
                ))
                await session.execute(sql_text(
                    f"CREATE UNIQUE INDEX {terms_table}_lexeme_idx ON {terms_table} (lexeme)"
                ))
                await session.execute(sql_text(
                    f"CREATE TABLE {stats_table} AS "
                    f"SELECT COUNT(*) AS n, AVG(dl) AS avgdl FROM ("
                    f"SELECT (SELECT COALESCE(SUM(array_length(t.positions, 1)), 0) "
                    f"FROM unnest({tsv_column}) t) AS dl FROM {collection_name}) docs"
                ))
                await session.commit()

        return True

    async def is_index_existed(self, collection_name: str) -> bool:
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            check_sql = sql_text(
                "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = :index_name)"
            )
            result = await session.execute(check_sql, {"index_name": index_name})
            return bool(result.scalar_one())

    async def create_vector_index(self, collection_name: str,
                                  index_type: str = PgVectorIndexTypeEnums.HNSW.value):

        is_index_existed = await self.is_index_existed(collection_name=collection_name)
        if is_index_existed:
            return False

        async with self.db_client() as session:
            async with session.begin():
                count_sql = sql_text(f"SELECT COUNT(*) FROM {collection_name}")
                result = await session.execute(count_sql)
                records_count = result.scalar_one()

                if records_count < self.index_threshold:
                    return False

                index_name = self.default_index_name(collection_name)
                create_idx_sql = sql_text(
                    f"CREATE INDEX {index_name} ON {collection_name} "
                    f"USING {index_type} ({PgVectorTableSchemeEnums.VECTOR.value} {self.distance_method})"
                )

                await session.execute(create_idx_sql)
                await session.commit()

        return True

    async def create_collection(self, collection_name: str, embedding_size: int,
                                do_reset: bool = False):

        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)

        if not await self.is_collection_existed(collection_name=collection_name):
            async with self.db_client() as session:
                async with session.begin():
                    create_sql = sql_text(
                        f"CREATE TABLE {collection_name} ("
                        f"{PgVectorTableSchemeEnums.ID.value} bigserial PRIMARY KEY,"
                        f"{PgVectorTableSchemeEnums.TEXT.value} text, "
                        f"{PgVectorTableSchemeEnums.VECTOR.value} vector({embedding_size}), "
                        f"{PgVectorTableSchemeEnums.METADATA.value} jsonb DEFAULT " + "'{}', " +
                        f"{PgVectorTableSchemeEnums.CHUNK_ID.value} integer, "
                        f"{PgVectorTableSchemeEnums.TEXT_TSV.value} tsvector GENERATED ALWAYS AS "
                        f"(to_tsvector('{self.ts_config}', "
                        f"{normalize_arabic_sql(PgVectorTableSchemeEnums.TEXT.value)})) STORED, "
                        f"FOREIGN KEY ({PgVectorTableSchemeEnums.CHUNK_ID.value}) REFERENCES chunks(chunk_id) ON DELETE CASCADE"
                        ")"
                    )
                    await session.execute(create_sql)

                    create_tsv_idx_sql = sql_text(
                        f"CREATE INDEX {self.default_tsv_index_name(collection_name)} "
                        f"ON {collection_name} USING gin ({PgVectorTableSchemeEnums.TEXT_TSV.value})"
                    )
                    await session.execute(create_tsv_idx_sql)
                    await session.commit()
            return True

        return False

    async def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, record_id: str = None):

        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(
                f"Can not insert new record to non-existed collection: {collection_name}")
            return False

        vector_str = "[" + ",".join([str(v) for v in vector]) + "]"
        metadata_json = json.dumps(metadata) if metadata else "{}"

        async with self.db_client() as session:
            async with session.begin():
                insert_sql = sql_text(
                    f"INSERT INTO {collection_name} (text, vector, metadata, chunk_id) "
                    f"VALUES (:text, :vector, :metadata, :chunk_id)"
                )
                await session.execute(insert_sql, {
                    "text": text,
                    "vector": vector_str,
                    "metadata": metadata_json,
                    "chunk_id": record_id,
                })
                await session.commit()

        return True

    async def insert_many(self, collection_name: str, texts: list, vectors: list,
                          metadata: list = None, record_ids: list = None,
                          batch_size: int = 50):

        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(
                f"Can not insert records to non-existed collection: {collection_name}")
            return False

        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = [None] * len(texts)

        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(texts), batch_size):
                    batch_end = i + batch_size

                    batch_texts = texts[i:batch_end]
                    batch_vectors = vectors[i:batch_end]
                    batch_metadata = metadata[i:batch_end]
                    batch_record_ids = record_ids[i:batch_end]

                    values = []
                    for x in range(len(batch_texts)):
                        _text = batch_texts[x]
                        _vector = batch_vectors[x]
                        _metadata = batch_metadata[x]
                        _record_id = batch_record_ids[x]

                        metadata_json = json.dumps(_metadata) if _metadata else "{}"

                        values.append({
                            "text": _text,
                            "vector": "[" + ",".join([str(v) for v in _vector]) + "]",
                            "metadata": metadata_json,
                            "chunk_id": _record_id,
                        })

                    batch_insert_sql = sql_text(
                        f"INSERT INTO {collection_name} "
                        f"(text, vector, metadata, chunk_id) "
                        f"VALUES (:text, :vector, :metadata, :chunk_id)"
                    )

                    await session.execute(batch_insert_sql, values)

                await session.commit()

        return True

    async def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):

        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(
                f"Can not search in non-existed collection: {collection_name}")
            return None

        vector_str = "[" + ",".join([str(v) for v in vector]) + "]"

        async with self.db_client() as session:
            search_sql = sql_text(
                f"SELECT text as text, 1 - (vector <=> :vector) as score"
                f" FROM {collection_name}"
                " ORDER BY score DESC "
                f"LIMIT {limit}"
            )
            result = await session.execute(search_sql, {"vector": vector_str})
            records = result.fetchall()

        if not records or len(records) == 0:
            return None

        return [
            RetrievedDocument(text=record.text, score=record.score)
            for record in records
        ]

    async def hybrid_search(self, collection_name: str, vector: list, query_text: str,
                            limit: int = 5, one_per_parent: bool = True):

        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(
                f"Can not search in non-existed collection: {collection_name}")
            return None

        vector_str = "[" + ",".join([str(v) for v in vector]) + "]"
        candidates = PgVectorHybridSearchEnums.CANDIDATES.value
        tsv_column = PgVectorTableSchemeEnums.TEXT_TSV.value
        vector_column = PgVectorTableSchemeEnums.VECTOR.value
        bm25_k1 = PgVectorHybridSearchEnums.BM25_K1.value
        bm25_b = PgVectorHybridSearchEnums.BM25_B.value
        bm25_terms_table = self.default_bm25_terms_table(collection_name)
        bm25_stats_table = self.default_bm25_stats_table(collection_name)

        if not await self.is_keyword_stats_existed(collection_name=collection_name):
            await self.refresh_keyword_stats(collection_name=collection_name)

        hybrid_sql = sql_text(
            f"""
            WITH semantic AS (
                SELECT chunk_id, 1 - ({vector_column} <=> :vector) AS semantic_score
                FROM {collection_name}
                ORDER BY {vector_column} <=> :vector
                LIMIT :candidates
            ),
            semantic_ranked AS (
                SELECT chunk_id, semantic_score,
                       RANK() OVER (ORDER BY semantic_score DESC) AS semantic_rank
                FROM semantic
            ),
            query_terms AS (
                SELECT DISTINCT lexeme
                FROM unnest(to_tsvector('{self.ts_config}',
                     {normalize_arabic_sql("replace(:query_text, ' or ', ' ')")}))
            ),
            keyword_docs AS (
                SELECT chunk_id, {tsv_column} AS tsv
                FROM {collection_name},
                     websearch_to_tsquery('{self.ts_config}', {normalize_arabic_sql(':query_text')}) AS query
                WHERE {tsv_column} @@ query
            ),
            keyword_terms AS (
                SELECT d.chunk_id,
                       array_length(t.positions, 1) AS tf,
                       (SELECT SUM(array_length(a.positions, 1)) FROM unnest(d.tsv) a) AS dl,
                       t.lexeme
                FROM keyword_docs d, unnest(d.tsv) t
                WHERE t.lexeme IN (SELECT lexeme FROM query_terms)
            ),
            keyword_scored AS (
                SELECT kt.chunk_id,
                       SUM(
                           ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
                           * kt.tf * ({bm25_k1} + 1)
                           / (kt.tf + {bm25_k1} * (1 - {bm25_b} + {bm25_b} * kt.dl / s.avgdl))
                       ) AS keyword_score
                FROM keyword_terms kt
                JOIN {bm25_terms_table} df ON df.lexeme = kt.lexeme
                CROSS JOIN {bm25_stats_table} s
                GROUP BY kt.chunk_id
            ),
            keyword AS (
                SELECT chunk_id, keyword_score,
                       RANK() OVER (ORDER BY keyword_score DESC) AS keyword_rank
                FROM keyword_scored
                ORDER BY keyword_score DESC, chunk_id
                LIMIT :candidates
            ),
            fused AS (
                SELECT COALESCE(s.chunk_id, k.chunk_id) AS chunk_id,
                       :semantic_weight * COALESCE(1.0 / (:rrf_k + s.semantic_rank), 0)
                         + :keyword_weight * COALESCE(1.0 / (:rrf_k + k.keyword_rank), 0) AS score,
                       s.semantic_score, s.semantic_rank,
                       k.keyword_score, k.keyword_rank
                FROM semantic_ranked s
                FULL OUTER JOIN keyword k ON s.chunk_id = k.chunk_id
            ),
            best_child AS (
                SELECT f.*, c.chunk_text, c.chunk_parent_id,
                       ROW_NUMBER() OVER (PARTITION BY c.chunk_parent_id ORDER BY f.score DESC, f.chunk_id) AS child_rank
                FROM fused f
                JOIN chunks c ON c.chunk_id = f.chunk_id
            )
            SELECT p.parent_id, p.parent_text, p.parent_metadata,
                   b.score, b.chunk_id, b.chunk_text,
                   b.semantic_score, b.semantic_rank, b.keyword_score, b.keyword_rank,
                   (SELECT array_agg(w.chunk_id ORDER BY w.chunk_order)
                    FROM chunks w WHERE w.chunk_parent_id = b.chunk_parent_id) AS children_ids,
                   (SELECT array_agg(w.chunk_text ORDER BY w.chunk_order)
                    FROM chunks w WHERE w.chunk_parent_id = b.chunk_parent_id) AS children_texts
            FROM best_child b
            JOIN parents p ON p.parent_id = b.chunk_parent_id
            WHERE b.child_rank <= :children_per_parent
            ORDER BY b.score DESC, p.parent_id
            LIMIT :limit
            """
        )

        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(
                    f"SET LOCAL hnsw.ef_search = {PgVectorHybridSearchEnums.EF_SEARCH.value}"))
                result = await session.execute(hybrid_sql, {
                    "vector": vector_str,
                    "query_text": query_text,
                    "candidates": candidates,
                    "rrf_k": PgVectorHybridSearchEnums.RRF_K.value,
                    "semantic_weight": PgVectorHybridSearchEnums.SEMANTIC_WEIGHT.value,
                    "keyword_weight": PgVectorHybridSearchEnums.KEYWORD_WEIGHT.value,
                    "limit": limit,
                    "children_per_parent": 1 if one_per_parent else candidates,
                })
                records = result.fetchall()

        if not records or len(records) == 0:
            return None

        return [
            RetrievedDocument(
                text=record.parent_text,
                score=float(record.score),
                metadata={
                    **(record.parent_metadata or {}),
                    "parent_id": record.parent_id,
                    "matched_chunk_id": record.chunk_id,
                    "matched_chunk_text": record.chunk_text,
                    "semantic_rank": record.semantic_rank,
                    "semantic_score": float(record.semantic_score) if record.semantic_score is not None else None,
                    "keyword_rank": record.keyword_rank,
                    "keyword_score": float(record.keyword_score) if record.keyword_score is not None else None,
                    "children_ids": list(record.children_ids or []),
                    "children_texts": list(record.children_texts or []),
                },
            )
            for record in records
        ]
