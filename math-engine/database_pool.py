"""
Database Connection Pool and Performance Monitoring
Provides optimized database connections with pooling and query performance tracking.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection
import structlog
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_hash: str
    query_template: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_executed: Optional[datetime] = None
    
    def update(self, execution_time: float):
        """Update metrics with new execution time."""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_executed = datetime.utcnow()


class DatabasePool:
    """Optimized database connection pool with performance monitoring."""
    
    def __init__(self, database_url: str, min_size: int = 5, max_size: int = 20):
        """Initialize database pool."""
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[Pool] = None
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.slow_query_threshold = 1.0  # seconds
        
    async def initialize(self):
        """Initialize the connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=30,
                server_settings={
                    'application_name': 'ai_math_tutor_engine',
                    'jit': 'off',  # Disable JIT for better performance on small queries
                }
            )
            logger.info("Database pool initialized", 
                       min_size=self.min_size, 
                       max_size=self.max_size)
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
    
    def _get_query_hash(self, query: str) -> str:
        """Generate a hash for query template (removing parameters)."""
        import hashlib
        import re
        
        # Normalize query by removing parameters and extra whitespace
        normalized = re.sub(r'\$\d+', '?', query)
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _record_query_metrics(self, query: str, execution_time: float):
        """Record query execution metrics."""
        query_hash = self._get_query_hash(query)
        
        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                query_template=query[:200] + "..." if len(query) > 200 else query
            )
        
        self.query_metrics[query_hash].update(execution_time)
        
        # Log slow queries
        if execution_time > self.slow_query_threshold:
            logger.warning("Slow query detected",
                          query_hash=query_hash,
                          execution_time=execution_time,
                          query_template=self.query_metrics[query_hash].query_template)
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a database connection from the pool."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the result."""
        start_time = time.time()
        
        try:
            async with self.acquire() as conn:
                result = await conn.execute(query, *args)
                
            execution_time = time.time() - start_time
            self._record_query_metrics(query, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Query execution failed",
                        query=query[:100],
                        execution_time=execution_time,
                        error=str(e))
            raise
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch multiple rows from a query."""
        start_time = time.time()
        
        try:
            async with self.acquire() as conn:
                result = await conn.fetch(query, *args)
                
            execution_time = time.time() - start_time
            self._record_query_metrics(query, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Query fetch failed",
                        query=query[:100],
                        execution_time=execution_time,
                        error=str(e))
            raise
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row from a query."""
        start_time = time.time()
        
        try:
            async with self.acquire() as conn:
                result = await conn.fetchrow(query, *args)
                
            execution_time = time.time() - start_time
            self._record_query_metrics(query, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Query fetchrow failed",
                        query=query[:100],
                        execution_time=execution_time,
                        error=str(e))
            raise
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value from a query."""
        start_time = time.time()
        
        try:
            async with self.acquire() as conn:
                result = await conn.fetchval(query, *args)
                
            execution_time = time.time() - start_time
            self._record_query_metrics(query, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Query fetchval failed",
                        query=query[:100],
                        execution_time=execution_time,
                        error=str(e))
            raise
    
    async def transaction(self):
        """Start a database transaction."""
        if not self.pool:
            await self.initialize()
        
        return self.pool.acquire()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self.pool:
            return {"status": "not_initialized"}
        
        return {
            "size": self.pool.get_size(),
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool.get_max_size(),
            "idle_size": self.pool.get_idle_size(),
            "status": "active"
        }
    
    def get_query_metrics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get query performance metrics."""
        sorted_metrics = sorted(
            self.query_metrics.values(),
            key=lambda x: x.avg_time,
            reverse=True
        )
        
        return [
            {
                "query_hash": metric.query_hash,
                "query_template": metric.query_template,
                "execution_count": metric.execution_count,
                "total_time": round(metric.total_time, 4),
                "avg_time": round(metric.avg_time, 4),
                "min_time": round(metric.min_time, 4),
                "max_time": round(metric.max_time, 4),
                "last_executed": metric.last_executed.isoformat() if metric.last_executed else None
            }
            for metric in sorted_metrics[:limit]
        ]
    
    def get_slow_queries(self, threshold: float = None) -> List[Dict[str, Any]]:
        """Get queries that exceed the slow query threshold."""
        if threshold is None:
            threshold = self.slow_query_threshold
        
        slow_queries = [
            metric for metric in self.query_metrics.values()
            if metric.avg_time > threshold
        ]
        
        return [
            {
                "query_hash": metric.query_hash,
                "query_template": metric.query_template,
                "avg_time": round(metric.avg_time, 4),
                "execution_count": metric.execution_count,
                "total_time": round(metric.total_time, 4)
            }
            for metric in sorted(slow_queries, key=lambda x: x.avg_time, reverse=True)
        ]
    
    async def optimize_queries(self):
        """Run query optimization analysis."""
        try:
            # Get database statistics
            stats_query = """
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables
            ORDER BY n_live_tup DESC;
            """
            
            table_stats = await self.fetch(stats_query)
            
            # Check for tables that need analysis
            needs_analysis = []
            for row in table_stats:
                if (row['last_analyze'] is None or 
                    row['last_analyze'] < datetime.utcnow() - timedelta(days=7)):
                    needs_analysis.append(row['tablename'])
            
            # Run ANALYZE on tables that need it
            if needs_analysis:
                for table in needs_analysis:
                    await self.execute(f"ANALYZE {table}")
                    logger.info(f"Analyzed table: {table}")
            
            # Get index usage statistics
            index_usage_query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_tup_read,
                idx_tup_fetch,
                idx_scan
            FROM pg_stat_user_indexes
            WHERE idx_scan = 0
            ORDER BY schemaname, tablename;
            """
            
            unused_indexes = await self.fetch(index_usage_query)
            
            if unused_indexes:
                logger.warning("Found unused indexes",
                              count=len(unused_indexes),
                              indexes=[f"{row['schemaname']}.{row['indexname']}" 
                                     for row in unused_indexes])
            
            return {
                "tables_analyzed": len(needs_analysis),
                "unused_indexes": len(unused_indexes),
                "table_stats": [dict(row) for row in table_stats],
                "unused_indexes_list": [dict(row) for row in unused_indexes]
            }
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            raise


class OptimizedQueries:
    """Collection of optimized queries for common operations."""
    
    def __init__(self, db_pool: DatabasePool):
        self.db = db_pool
    
    async def get_user_performance_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive user performance summary using materialized view."""
        query = """
        SELECT * FROM user_performance_summary 
        WHERE user_id = $1
        """
        
        row = await self.db.fetchrow(query, user_id)
        return dict(row) if row else None
    
    async def get_user_learning_recommendations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get personalized learning recommendations."""
        query = "SELECT * FROM get_user_learning_recommendations($1, $2)"
        
        rows = await self.db.fetch(query, user_id, limit)
        return [dict(row) for row in rows]
    
    async def get_problem_recommendations(self, user_id: str, domain: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get problem recommendations based on user performance."""
        query = "SELECT * FROM get_problem_recommendations($1, $2, $3)"
        
        rows = await self.db.fetch(query, user_id, domain, limit)
        return [dict(row) for row in rows]
    
    async def get_user_progress_by_topic(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user progress across all topics with optimized query."""
        query = """
        SELECT 
            lp.topic,
            lp.mastery_level,
            lp.practice_count,
            lp.total_time_spent,
            lp.last_practiced,
            COUNT(pa.id) as recent_attempts,
            COUNT(pa.id) FILTER (WHERE pa.is_correct) as recent_correct
        FROM learning_progress lp
        LEFT JOIN problem_attempts pa ON pa.user_id = lp.user_id 
            AND pa.attempt_timestamp > NOW() - INTERVAL '7 days'
        LEFT JOIN problems p ON pa.problem_id = p.id AND p.domain = lp.topic
        WHERE lp.user_id = $1
        GROUP BY lp.topic, lp.mastery_level, lp.practice_count, 
                 lp.total_time_spent, lp.last_practiced
        ORDER BY lp.mastery_level ASC, lp.last_practiced DESC
        """
        
        rows = await self.db.fetch(query, user_id)
        return [dict(row) for row in rows]
    
    async def get_quiz_performance_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get quiz performance analytics with optimized aggregation."""
        query = """
        WITH quiz_stats AS (
            SELECT 
                topic,
                difficulty_level,
                COUNT(*) as quiz_count,
                AVG(points_earned::decimal / NULLIF(total_points, 0)) as avg_score,
                AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration
            FROM quiz_sessions
            WHERE user_id = $1 
                AND status = 'completed'
                AND completed_at > NOW() - INTERVAL '%s days'
            GROUP BY topic, difficulty_level
        ),
        overall_stats AS (
            SELECT 
                COUNT(*) as total_quizzes,
                AVG(points_earned::decimal / NULLIF(total_points, 0)) as overall_avg_score,
                MAX(points_earned::decimal / NULLIF(total_points, 0)) as best_score,
                COUNT(*) FILTER (WHERE points_earned::decimal / NULLIF(total_points, 0) >= 0.8) as high_score_count
            FROM quiz_sessions
            WHERE user_id = $1 
                AND status = 'completed'
                AND completed_at > NOW() - INTERVAL '%s days'
        )
        SELECT 
            json_build_object(
                'overall', row_to_json(os.*),
                'by_topic', json_agg(row_to_json(qs.*))
            ) as analytics
        FROM overall_stats os
        CROSS JOIN quiz_stats qs
        GROUP BY os.total_quizzes, os.overall_avg_score, os.best_score, os.high_score_count
        """ % (days, days)
        
        result = await self.db.fetchval(query, user_id)
        return result if result else {"overall": {}, "by_topic": []}
    
    async def get_problem_difficulty_insights(self, domain: str = None) -> List[Dict[str, Any]]:
        """Get problem difficulty insights using materialized view."""
        if domain:
            query = """
            SELECT * FROM problem_difficulty_analysis 
            WHERE domain = $1
            ORDER BY success_rate ASC, total_attempts DESC
            """
            rows = await self.db.fetch(query, domain)
        else:
            query = """
            SELECT * FROM problem_difficulty_analysis 
            ORDER BY success_rate ASC, total_attempts DESC
            LIMIT 50
            """
            rows = await self.db.fetch(query)
        
        return [dict(row) for row in rows]
    
    async def batch_update_user_progress(self, progress_updates: List[Dict[str, Any]]) -> int:
        """Efficiently batch update user progress using UPSERT."""
        if not progress_updates:
            return 0
        
        query = """
        INSERT INTO learning_progress (user_id, topic, mastery_level, practice_count, total_time_spent, last_practiced)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (user_id, topic)
        DO UPDATE SET
            mastery_level = EXCLUDED.mastery_level,
            practice_count = learning_progress.practice_count + EXCLUDED.practice_count,
            total_time_spent = learning_progress.total_time_spent + EXCLUDED.total_time_spent,
            last_practiced = EXCLUDED.last_practiced
        """
        
        async with self.db.transaction() as conn:
            async with conn.transaction():
                updated_count = 0
                for update in progress_updates:
                    await conn.execute(
                        query,
                        update['user_id'],
                        update['topic'],
                        update['mastery_level'],
                        update.get('practice_count', 1),
                        update.get('total_time_spent', 0),
                        update.get('last_practiced', datetime.utcnow())
                    )
                    updated_count += 1
        
        return updated_count


# Global database pool instance
db_pool = DatabasePool("postgresql://postgres:password@localhost:5432/ai_math_tutor")
optimized_queries = OptimizedQueries(db_pool)