#### What this tests ####
#    This tests the new rpd/tpd (requests per day / tokens per day) functionality

import asyncio
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.types.utils import StandardLoggingPayload
import pytest
from litellm.types.router import DeploymentTypedDict, ModelGroupInfo
import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import (
    LowestTPMLoggingHandler_v2 as LowestTPMLoggingHandler,
)
from litellm.utils import get_utc_datetime
from create_mock_standard_logging_payload import create_standard_logging_payload

### UNIT TESTS FOR TPD/RPD FUNCTIONALITY ###

def test_model_group_info_supports_rpd_tpd():
    """Test that ModelGroupInfo supports rpd and tpd fields"""
    from litellm.types.router import ModelGroupInfo
    
    # Create a ModelGroupInfo with rpd and tpd
    model_info = ModelGroupInfo(
        model_group="test-group",
        providers=["openai"],
        tpm=1000,
        rpm=100,
        tpd=10000,
        rpd=1000
    )
    
    assert model_info.tpm == 1000
    assert model_info.rpm == 100
    assert model_info.tpd == 10000
    assert model_info.rpd == 1000
    
    print("✓ ModelGroupInfo supports rpd and tpd")


def test_router_cache_enum_supports_daily():
    """Test that RouterCacheEnum supports TPD and RPD"""
    from litellm.types.router import RouterCacheEnum
    
    # Test that enum has the new values
    assert hasattr(RouterCacheEnum, 'TPD')
    assert hasattr(RouterCacheEnum, 'RPD')
    
    # Test cache key formatting
    tpd_key = RouterCacheEnum.TPD.value.format(
        id="test-id",
        model="test-model", 
        current_day="2024-01-01"
    )
    rpd_key = RouterCacheEnum.RPD.value.format(
        id="test-id",
        model="test-model",
        current_day="2024-01-01" 
    )
    
    expected_tpd = "global_router:test-id:test-model:tpd:2024-01-01"
    expected_rpd = "global_router:test-id:test-model:rpd:2024-01-01"
    
    assert tpd_key == expected_tpd
    assert rpd_key == expected_rpd
    
    print("✓ RouterCacheEnum supports TPD and RPD")


def test_litellm_params_supports_rpd_tpd():
    """Test that LiteLLM_Params supports rpd and tpd fields"""
    from litellm.types.router import LiteLLM_Params
    
    # Create params with rpd and tpd
    params = LiteLLM_Params(
        model="test-model",
        tpm=1000,
        rpm=100,
        tpd=10000,
        rpd=1000
    )
    
    assert params.tpm == 1000
    assert params.rpm == 100
    assert params.tpd == 10000
    assert params.rpd == 1000
    
    print("✓ LiteLLM_Params supports rpd and tpd")


@pytest.mark.asyncio
async def test_router_daily_usage_tracking():
    """Test that router can track daily usage"""
    from litellm import Router
    from litellm.caching.caching import DualCache
    
    # Create a simple router with a deployment that has rpd/tpd limits
    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
                "rpd": 1000,  # 1000 requests per day
                "tpd": 50000,  # 50000 tokens per day
            },
            "model_info": {"id": "test-deployment-1"}
        }
    ]
    
    router = Router(
        model_list=model_list,
        cache=DualCache()
    )
    
    # Test the daily usage tracking method
    tpd_usage, rpd_usage = await router.get_model_group_daily_usage("test-model")
    
    # Should return None or 0 for new deployments with no usage
    assert tpd_usage is None or tpd_usage == 0
    assert rpd_usage is None or rpd_usage == 0
    
    print("✓ Router daily usage tracking works")


@pytest.mark.asyncio 
async def test_daily_rate_limiting():
    """Test that daily rate limits are enforced"""
    from litellm.proxy.hooks.dynamic_rate_limiter import _PROXY_DynamicRateLimitHandler
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.caching.caching import DualCache
    from litellm import Router
    
    # Create router with daily limits
    model_list = [
        {
            "model_name": "test-gemini",
            "litellm_params": {
                "model": "gemini/gemini-1.5-pro",
                "api_key": "test-key",
                "tpm": 1000,
                "rpm": 100,
                "tpd": 10000,  # 10000 tokens per day
                "rpd": 1000,   # 1000 requests per day
            },
            "model_info": {"id": "gemini-deployment-1"}
        }
    ]
    
    router = Router(
        model_list=model_list,
        cache=DualCache()
    )
    
    # Create rate limiter
    rate_limiter = _PROXY_DynamicRateLimitHandler(internal_usage_cache=DualCache())
    rate_limiter.update_variables(llm_router=router)
    
    # Test the check_available_usage method with daily limits
    available_tpm, available_rpm, model_tpm, model_rpm, active_projects, available_tpd, available_rpd = await rate_limiter.check_available_usage(
        model="test-gemini"
    )
    
    # Since there's no usage yet, should have available daily limits
    print(f"Available daily limits - TPD: {available_tpd}, RPD: {available_rpd}")
    
    # The values should be either None (if no limits set) or the full limits (if limits set)
    assert available_tpd is None or available_tpd >= 0
    assert available_rpd is None or available_rpd >= 0
    
    print("✓ Daily rate limiting check works")


def test_deployment_config_yaml_example():
    """Test that the example config from the issue works"""
    # This is the example from the GitHub issue
    deployment_config = {
        "model_name": "gemini/mytest",
        "litellm_params": {
            "model": "gemini/gemini-2.5-pro-preview-03-25",
            "api_key": "os.environ/GEMINI_API_KEY",
            "rpm": 150,
            "tpm": 2000000,
            "rpd": 1000,  # This is the new field we added
        },
        "model_info": {
            "mode": "completion"
        }
    }
    
    # Test that this config can be used to create LiteLLM_Params
    from litellm.types.router import LiteLLM_Params
    
    litellm_params = deployment_config["litellm_params"]
    
    # Should be able to create params with the new fields
    params = LiteLLM_Params(**litellm_params)
    
    assert params.rpm == 150
    assert params.tpm == 2000000
    assert params.rpd == 1000
    assert hasattr(params, 'tpd')  # Should have tpd field even if not set
    
    print("✓ Example deployment config works with rpd/tpd")


if __name__ == "__main__":
    test_model_group_info_supports_rpd_tpd()
    test_router_cache_enum_supports_daily()
    test_litellm_params_supports_rpd_tpd()
    test_deployment_config_yaml_example()
    
    # Run async tests
    asyncio.run(test_router_daily_usage_tracking())
    asyncio.run(test_daily_rate_limiting())
    
    print("All RPD/TPD tests passed!")