#!/usr/bin/env python3
"""
Simple test to verify the RPD/TPD feature works
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_basic_imports():
    """Test that our new fields can be imported and used"""
    from litellm.types.router import GenericLiteLLMParams, LiteLLM_Params, ModelGroupInfo, RouterCacheEnum
    
    # Test that GenericLiteLLMParams can be created with new fields
    params = GenericLiteLLMParams(
        tpm=100,
        rpm=10,
        tpd=10000,
        rpd=1000
    )
    assert params.tpm == 100
    assert params.rpm == 10
    assert params.tpd == 10000
    assert params.rpd == 1000
    print("✓ GenericLiteLLMParams supports rpd and tpd")
    
    # Test that LiteLLM_Params can be created with new fields
    litellm_params = LiteLLM_Params(
        model="test-model",
        tpm=100,
        rpm=10,
        tpd=10000,
        rpd=1000
    )
    assert litellm_params.tpm == 100
    assert litellm_params.rpm == 10
    assert litellm_params.tpd == 10000
    assert litellm_params.rpd == 1000
    print("✓ LiteLLM_Params supports rpd and tpd")
    
    # Test that ModelGroupInfo can be created with new fields
    model_info = ModelGroupInfo(
        model_group="test-group",
        providers=["openai"],
        tpm=100,
        rpm=10,
        tpd=10000,
        rpd=1000
    )
    assert model_info.tpm == 100
    assert model_info.rpm == 10
    assert model_info.tpd == 10000
    assert model_info.rpd == 1000
    print("✓ ModelGroupInfo supports rpd and tpd")
    
    # Test that RouterCacheEnum has new values
    assert hasattr(RouterCacheEnum, 'TPD')
    assert hasattr(RouterCacheEnum, 'RPD')
    
    # Test formatting the cache keys
    cache_key_tpd = RouterCacheEnum.TPD.value.format(
        id="test-id", 
        model="test-model", 
        current_day="2024-01-01"
    )
    cache_key_rpd = RouterCacheEnum.RPD.value.format(
        id="test-id", 
        model="test-model", 
        current_day="2024-01-01"
    )
    
    expected_tpd = "global_router:test-id:test-model:tpd:2024-01-01"
    expected_rpd = "global_router:test-id:test-model:rpd:2024-01-01"
    
    assert cache_key_tpd == expected_tpd, f"Expected {expected_tpd}, got {cache_key_tpd}"
    assert cache_key_rpd == expected_rpd, f"Expected {expected_rpd}, got {cache_key_rpd}"
    print("✓ RouterCacheEnum TPD and RPD work correctly")
    
    print("All basic tests passed!")

if __name__ == "__main__":
    test_basic_imports()