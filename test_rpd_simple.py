#!/usr/bin/env python3
"""
Simple test to verify the RPD/TPD feature works - importing only what we need
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Import without going through the main litellm module
if True:  # scope block
    import enum
    from typing import Optional, List
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict

    # Import the router types directly
    sys.path.insert(0, os.path.join(os.path.abspath('.'), 'litellm'))
    from types.router import RouterCacheEnum
    
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
    
    print("RouterCacheEnum test passed!")