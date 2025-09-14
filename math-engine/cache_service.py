"""
Redis Cache Service for AI Math Tutor
Provides caching functionality for problem solutions, AI explanations, and visualizations.
"""

import json
import hashlib
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import redis.asyncio as redis
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class CacheMetrics(BaseModel):
    """Cache performance metrics."""
    hit_count: int = 0
    miss_count: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0
    cache_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hit_count / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class CacheService:
    """Redis-based caching service with performance monitoring."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize cache service with Redis connection."""
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()
        
        # Cache TTL settings (in seconds)
        self.ttl_settings = {
            "problem_solution": 3600,      # 1 hour
            "ai_explanation": 7200,        # 2 hours
            "visualization": 1800,         # 30 minutes
            "user_session": 86400,         # 24 hours
            "quiz_data": 3600,            # 1 hour
            "user_progress": 1800,        # 30 minutes
        }
    
    async def connect(self):
        """Establish Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache service connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis cache service disconnected")
    
    def _generate_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate a consistent cache key from data."""
        # Sort the data to ensure consistent key generation
        sorted_data = json.dumps(data, sort_keys=True)
        data_hash = hashlib.md5(sorted_data.encode()).hexdigest()
        return f"{prefix}:{data_hash}"
    
    async def _record_metrics(self, operation: str, start_time: float, hit: bool):
        """Record cache operation metrics."""
        response_time = time.time() - start_time
        
        self.metrics.total_requests += 1
        if hit:
            self.metrics.hit_count += 1
        else:
            self.metrics.miss_count += 1
        
        # Update rolling average response time
        total_time = self.metrics.average_response_time * (self.metrics.total_requests - 1)
        self.metrics.average_response_time = (total_time + response_time) / self.metrics.total_requests
        
        logger.debug(f"Cache {operation}", 
                    hit=hit, 
                    response_time=response_time,
                    hit_rate=self.metrics.hit_rate)
    
    async def get(self, cache_type: str, key_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached data by type and key data."""
        if not self.redis_client:
            await self.connect()
        
        start_time = time.time()
        cache_key = self._generate_cache_key(cache_type, key_data)
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                await self._record_metrics("get", start_time, hit=True)
                return json.loads(cached_data)
            else:
                await self._record_metrics("get", start_time, hit=False)
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}", cache_key=cache_key)
            await self._record_metrics("get", start_time, hit=False)
            return None
    
    async def set(self, cache_type: str, key_data: Dict[str, Any], 
                  value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set cached data with optional TTL."""
        if not self.redis_client:
            await self.connect()
        
        start_time = time.time()
        cache_key = self._generate_cache_key(cache_type, key_data)
        
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.ttl_settings.get(cache_type, 3600)
        
        try:
            # Add metadata to cached value
            cached_value = {
                "data": value,
                "cached_at": datetime.utcnow().isoformat(),
                "cache_type": cache_type,
                "ttl": ttl
            }
            
            await self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(cached_value, default=str)
            )
            
            # Don't record set operations in hit/miss metrics
            response_time = time.time() - start_time
            logger.debug(f"Cached data", cache_key=cache_key, ttl=ttl, response_time=response_time)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}", cache_key=cache_key)
            return False
    
    async def delete(self, cache_type: str, key_data: Dict[str, Any]) -> bool:
        """Delete cached data."""
        if not self.redis_client:
            await self.connect()
        
        cache_key = self._generate_cache_key(cache_type, key_data)
        
        try:
            result = await self.redis_client.delete(cache_key)
            logger.debug(f"Deleted cache entry", cache_key=cache_key, existed=bool(result))
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}", cache_key=cache_key)
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching a pattern."""
        if not self.redis_client:
            await self.connect()
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted_count = await self.redis_client.delete(*keys)
                logger.info(f"Invalidated cache entries", pattern=pattern, count=deleted_count)
                return deleted_count
            return 0
            
        except Exception as e:
            logger.error(f"Cache pattern invalidation error: {e}", pattern=pattern)
            return 0
    
    async def get_metrics(self) -> CacheMetrics:
        """Get current cache performance metrics."""
        if self.redis_client:
            try:
                info = await self.redis_client.info("memory")
                self.metrics.cache_size = info.get("used_memory", 0)
            except Exception as e:
                logger.error(f"Error getting cache size: {e}")
        
        return self.metrics
    
    async def clear_all(self) -> bool:
        """Clear all cached data (use with caution)."""
        if not self.redis_client:
            await self.connect()
        
        try:
            await self.redis_client.flushdb()
            logger.warning("All cache data cleared")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False


# Specialized cache methods for different data types

class ProblemSolutionCache:
    """Specialized caching for problem solutions."""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
    
    async def get_solution(self, problem_text: str, domain: str) -> Optional[Dict[str, Any]]:
        """Get cached problem solution."""
        key_data = {
            "problem_text": problem_text,
            "domain": domain,
            "type": "solution"
        }
        
        cached_result = await self.cache.get("problem_solution", key_data)
        if cached_result:
            return cached_result.get("data")
        return None
    
    async def cache_solution(self, problem_text: str, domain: str, 
                           solution: Dict[str, Any]) -> bool:
        """Cache a problem solution."""
        key_data = {
            "problem_text": problem_text,
            "domain": domain,
            "type": "solution"
        }
        
        return await self.cache.set("problem_solution", key_data, solution)
    
    async def invalidate_user_solutions(self, user_id: str) -> int:
        """Invalidate cached solutions for a specific user."""
        pattern = f"problem_solution:*user_id*{user_id}*"
        return await self.cache.invalidate_pattern(pattern)


class AIExplanationCache:
    """Specialized caching for AI explanations."""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
    
    async def get_explanation(self, step_data: Dict[str, Any], 
                            user_level: str) -> Optional[Dict[str, Any]]:
        """Get cached AI explanation."""
        key_data = {
            "step_operation": step_data.get("operation", ""),
            "step_expression": step_data.get("mathematical_expression", ""),
            "user_level": user_level,
            "type": "explanation"
        }
        
        cached_result = await self.cache.get("ai_explanation", key_data)
        if cached_result:
            return cached_result.get("data")
        return None
    
    async def cache_explanation(self, step_data: Dict[str, Any], 
                              user_level: str, explanation: Dict[str, Any]) -> bool:
        """Cache an AI explanation."""
        key_data = {
            "step_operation": step_data.get("operation", ""),
            "step_expression": step_data.get("mathematical_expression", ""),
            "user_level": user_level,
            "type": "explanation"
        }
        
        return await self.cache.set("ai_explanation", key_data, explanation)


class VisualizationCache:
    """Specialized caching for visualizations."""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
    
    async def get_visualization(self, problem_data: Dict[str, Any], 
                              viz_type: str) -> Optional[Dict[str, Any]]:
        """Get cached visualization."""
        key_data = {
            "problem_text": problem_data.get("problem_text", ""),
            "domain": problem_data.get("domain", ""),
            "viz_type": viz_type,
            "type": "visualization"
        }
        
        cached_result = await self.cache.get("visualization", key_data)
        if cached_result:
            return cached_result.get("data")
        return None
    
    async def cache_visualization(self, problem_data: Dict[str, Any], 
                                viz_type: str, visualization: Dict[str, Any]) -> bool:
        """Cache a visualization."""
        key_data = {
            "problem_text": problem_data.get("problem_text", ""),
            "domain": problem_data.get("domain", ""),
            "viz_type": viz_type,
            "type": "visualization"
        }
        
        return await self.cache.set("visualization", key_data, visualization)


# Global cache service instance
cache_service = CacheService()
problem_cache = ProblemSolutionCache(cache_service)
ai_explanation_cache = AIExplanationCache(cache_service)
visualization_cache = VisualizationCache(cache_service)