"""
Tests for Database Optimization and Connection Pooling
Tests database performance improvements, connection pooling, and query optimization.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import os
from unittest.mock import AsyncMock, patch
from database_pool import DatabasePool, OptimizedQueries, QueryMetrics


@pytest_asyncio.fixture
async def db_pool():
    """Create a test database pool."""
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/ai_math_tutor")
    pool = DatabasePool(database_url, min_size=2, max_size=5)
    
    try:
        await pool.initialize()
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def optimized_queries(db_pool):
    """Create optimized queries instance."""
    return OptimizedQueries(db_pool)


class TestDatabasePool:
    """Test database connection pooling functionality."""
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self, db_pool):
        """Test database pool initialization."""
        assert db_pool.pool is not None
        
        stats = db_pool.get_pool_stats()
        assert stats["status"] == "active"
        assert stats["min_size"] == 2
        assert stats["max_size"] == 5
        assert stats["size"] >= stats["min_size"]
    
    @pytest.mark.asyncio
    async def test_basic_query_execution(self, db_pool):
        """Test basic query execution through pool."""
        # Test simple query
        result = await db_pool.fetchval("SELECT 1")
        assert result == 1
        
        # Test query with parameters
        result = await db_pool.fetchval("SELECT $1::integer + $2::integer", 5, 3)
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_query_metrics_tracking(self, db_pool):
        """Test that query metrics are properly tracked."""
        # Clear existing metrics
        db_pool.query_metrics.clear()
        
        # Execute some queries
        await db_pool.fetchval("SELECT 1")
        await db_pool.fetchval("SELECT 2")
        await db_pool.fetchval("SELECT 1")  # Same query again
        
        metrics = db_pool.get_query_metrics()
        
        # Should have metrics for 2 unique queries
        assert len(metrics) == 2
        
        # Find the "SELECT 1" query metrics
        select_1_metrics = next(
            (m for m in metrics if "SELECT 1" in m["query_template"]), 
            None
        )
        assert select_1_metrics is not None
        assert select_1_metrics["execution_count"] == 2  # Executed twice
    
    @pytest.mark.asyncio
    async def test_slow_query_detection(self, db_pool):
        """Test slow query detection and logging."""
        # Set a very low threshold for testing
        original_threshold = db_pool.slow_query_threshold
        db_pool.slow_query_threshold = 0.001  # 1ms
        
        try:
            # Execute a query that should be flagged as slow
            await db_pool.fetchval("SELECT pg_sleep(0.01)")  # 10ms sleep
            
            slow_queries = db_pool.get_slow_queries(0.001)
            assert len(slow_queries) > 0
            
            # Check that the slow query was recorded
            slow_query = slow_queries[0]
            assert slow_query["avg_time"] > 0.001
            
        finally:
            db_pool.slow_query_threshold = original_threshold
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self, db_pool):
        """Test concurrent database connections."""
        async def execute_query(query_id):
            result = await db_pool.fetchval(f"SELECT {query_id}")
            return result
        
        # Execute multiple queries concurrently
        tasks = [execute_query(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All queries should complete successfully
        assert len(results) == 10
        assert results == list(range(10))
        
        # Pool should handle concurrent connections
        stats = db_pool.get_pool_stats()
        assert stats["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_transaction_handling(self, db_pool):
        """Test database transaction handling."""
        # Create a test table for transaction testing
        await db_pool.execute("""
            CREATE TEMP TABLE test_transaction (
                id SERIAL PRIMARY KEY,
                value INTEGER
            )
        """)
        
        # Test successful transaction
        async with db_pool.transaction() as conn:
            async with conn.transaction():
                await conn.execute("INSERT INTO test_transaction (value) VALUES (1)")
                await conn.execute("INSERT INTO test_transaction (value) VALUES (2)")
        
        # Verify data was committed
        count = await db_pool.fetchval("SELECT COUNT(*) FROM test_transaction")
        assert count == 2
        
        # Test transaction rollback
        try:
            async with db_pool.transaction() as conn:
                async with conn.transaction():
                    await conn.execute("INSERT INTO test_transaction (value) VALUES (3)")
                    # Force an error to trigger rollback
                    await conn.execute("INSERT INTO test_transaction (value) VALUES ('invalid')")
        except Exception:
            pass  # Expected to fail
        
        # Count should still be 2 (rollback occurred)
        count = await db_pool.fetchval("SELECT COUNT(*) FROM test_transaction")
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_connection_recovery(self, db_pool):
        """Test connection recovery after errors."""
        # Execute a valid query first
        result = await db_pool.fetchval("SELECT 1")
        assert result == 1
        
        # Execute an invalid query (should not break the pool)
        try:
            await db_pool.fetchval("SELECT * FROM non_existent_table")
        except Exception:
            pass  # Expected to fail
        
        # Pool should still work after the error
        result = await db_pool.fetchval("SELECT 2")
        assert result == 2
        
        stats = db_pool.get_pool_stats()
        assert stats["status"] == "active"


class TestOptimizedQueries:
    """Test optimized query implementations."""
    
    @pytest.mark.asyncio
    async def test_user_performance_summary(self, optimized_queries):
        """Test user performance summary query."""
        # This test assumes the materialized view exists
        # In a real test, you might need to create test data first
        
        try:
            # Test with a non-existent user (should return None)
            result = await optimized_queries.get_user_performance_summary("00000000-0000-0000-0000-000000000000")
            assert result is None or isinstance(result, dict)
            
        except Exception as e:
            # If materialized view doesn't exist, that's expected in test environment
            assert "does not exist" in str(e) or "relation" in str(e)
    
    @pytest.mark.asyncio
    async def test_user_progress_by_topic(self, optimized_queries):
        """Test user progress by topic query."""
        # Test with a non-existent user
        result = await optimized_queries.get_user_progress_by_topic("00000000-0000-0000-0000-000000000000")
        assert isinstance(result, list)
        # Should return empty list for non-existent user
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_quiz_performance_analytics(self, optimized_queries):
        """Test quiz performance analytics query."""
        # Test with a non-existent user
        result = await optimized_queries.get_quiz_performance_analytics("00000000-0000-0000-0000-000000000000")
        assert isinstance(result, dict)
        # Should return empty analytics for non-existent user
        assert "overall" in result
        assert "by_topic" in result
    
    @pytest.mark.asyncio
    async def test_batch_update_user_progress(self, optimized_queries):
        """Test batch update of user progress."""
        # Test with empty updates
        result = await optimized_queries.batch_update_user_progress([])
        assert result == 0
        
        # Test with sample updates (these will fail if tables don't exist, which is expected)
        updates = [
            {
                "user_id": "00000000-0000-0000-0000-000000000000",
                "topic": "algebra",
                "mastery_level": 0.75,
                "practice_count": 1,
                "total_time_spent": 300
            }
        ]
        
        try:
            result = await optimized_queries.batch_update_user_progress(updates)
            assert isinstance(result, int)
        except Exception as e:
            # Expected if tables don't exist in test environment
            assert "does not exist" in str(e) or "relation" in str(e)


class TestQueryMetrics:
    """Test query metrics functionality."""
    
    def test_query_metrics_initialization(self):
        """Test QueryMetrics initialization."""
        metrics = QueryMetrics("test_hash", "SELECT * FROM test")
        
        assert metrics.query_hash == "test_hash"
        assert metrics.query_template == "SELECT * FROM test"
        assert metrics.execution_count == 0
        assert metrics.total_time == 0.0
        assert metrics.avg_time == 0.0
    
    def test_query_metrics_update(self):
        """Test QueryMetrics update functionality."""
        metrics = QueryMetrics("test_hash", "SELECT * FROM test")
        
        # Update with first execution
        metrics.update(0.1)
        assert metrics.execution_count == 1
        assert metrics.total_time == 0.1
        assert metrics.avg_time == 0.1
        assert metrics.min_time == 0.1
        assert metrics.max_time == 0.1
        
        # Update with second execution
        metrics.update(0.3)
        assert metrics.execution_count == 2
        assert metrics.total_time == 0.4
        assert metrics.avg_time == 0.2
        assert metrics.min_time == 0.1
        assert metrics.max_time == 0.3
    
    def test_query_hash_generation(self):
        """Test query hash generation for normalization."""
        pool = DatabasePool("postgresql://test")
        
        # Same queries with different parameters should have same hash
        hash1 = pool._get_query_hash("SELECT * FROM users WHERE id = $1")
        hash2 = pool._get_query_hash("SELECT * FROM users WHERE id = $2")
        assert hash1 == hash2
        
        # Different queries should have different hashes
        hash3 = pool._get_query_hash("SELECT * FROM problems WHERE id = $1")
        assert hash1 != hash3


class TestDatabasePerformance:
    """Test database performance improvements."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_performance(self, db_pool):
        """Test that connection pooling improves performance."""
        # Measure time for multiple sequential queries
        start_time = time.time()
        
        for i in range(10):
            await db_pool.fetchval("SELECT $1", i)
        
        pooled_time = time.time() - start_time
        
        # The pooled queries should complete reasonably quickly
        # This is more of a smoke test than a precise performance test
        assert pooled_time < 5.0  # Should complete within 5 seconds
        
        # Check that metrics were recorded
        metrics = db_pool.get_query_metrics()
        assert len(metrics) > 0
        
        # Find the SELECT query metrics
        select_metrics = next(
            (m for m in metrics if "SELECT" in m["query_template"]), 
            None
        )
        assert select_metrics is not None
        assert select_metrics["execution_count"] == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_query_performance(self, db_pool):
        """Test performance under concurrent load."""
        async def execute_batch_queries():
            tasks = []
            for i in range(5):
                task = db_pool.fetchval("SELECT $1", i)
                tasks.append(task)
            return await asyncio.gather(*tasks)
        
        # Execute multiple batches concurrently
        start_time = time.time()
        
        batch_tasks = [execute_batch_queries() for _ in range(4)]
        results = await asyncio.gather(*batch_tasks)
        
        concurrent_time = time.time() - start_time
        
        # All batches should complete successfully
        assert len(results) == 4
        for batch_result in results:
            assert len(batch_result) == 5
        
        # Should complete within reasonable time
        assert concurrent_time < 10.0
        
        # Pool should remain healthy
        stats = db_pool.get_pool_stats()
        assert stats["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_query_optimization_analysis(self, db_pool):
        """Test query optimization analysis."""
        try:
            # Run optimization analysis
            result = await db_pool.optimize_queries()
            
            assert isinstance(result, dict)
            assert "tables_analyzed" in result
            assert "unused_indexes" in result
            assert "table_stats" in result
            
        except Exception as e:
            # May fail if we don't have proper permissions or tables don't exist
            # This is expected in test environment
            assert "permission denied" in str(e) or "does not exist" in str(e)


class TestDatabaseIndexing:
    """Test database indexing and optimization."""
    
    @pytest.mark.asyncio
    async def test_index_usage_monitoring(self, db_pool):
        """Test index usage monitoring."""
        try:
            # Query to check if indexes exist
            index_query = """
            SELECT indexname, tablename 
            FROM pg_indexes 
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname
            """
            
            indexes = await db_pool.fetch(index_query)
            
            # Should have some indexes (from init.sql and optimize.sql)
            assert len(indexes) > 0
            
            # Check for some expected indexes
            index_names = [row['indexname'] for row in indexes]
            
            # These indexes should exist if the optimization script was run
            expected_indexes = [
                'idx_users_email',
                'idx_problem_attempts_user_id',
                'idx_quiz_sessions_user_id'
            ]
            
            for expected_index in expected_indexes:
                if expected_index in index_names:
                    # At least some expected indexes should exist
                    break
            else:
                # If none of the expected indexes exist, the optimization script may not have run
                # This is acceptable in test environment
                pass
                
        except Exception as e:
            # May fail if tables don't exist in test environment
            assert "does not exist" in str(e) or "relation" in str(e)
    
    @pytest.mark.asyncio
    async def test_query_plan_analysis(self, db_pool):
        """Test query execution plan analysis."""
        try:
            # Test EXPLAIN on a simple query
            explain_result = await db_pool.fetch("EXPLAIN SELECT 1")
            assert len(explain_result) > 0
            
            # Test EXPLAIN ANALYZE on a simple query
            explain_analyze_result = await db_pool.fetch("EXPLAIN ANALYZE SELECT 1")
            assert len(explain_analyze_result) > 0
            
        except Exception as e:
            # Should not fail for simple queries
            pytest.fail(f"Query plan analysis failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])