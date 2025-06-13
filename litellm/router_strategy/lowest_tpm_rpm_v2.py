#### What this does ####
#   identifies lowest tpm deployment
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm import token_counter
from litellm._logging import verbose_logger, verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.router import RouterErrors
from litellm.types.utils import LiteLLMPydanticObjectBase, StandardLoggingPayload
from litellm.utils import get_utc_datetime, print_verbose

from .base_routing_strategy import BaseRoutingStrategy

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class RoutingArgs(LiteLLMPydanticObjectBase):
    ttl: int = 1 * 60  # 1min (RPM/TPM expire key)


class LowestTPMLoggingHandler_v2(BaseRoutingStrategy, CustomLogger):
    """
    Updated version of TPM/RPM Logging.

    Meant to work across instances.

    Caches individual models, not model_groups

    Uses batch get (redis.mget)

    Increments tpm/rpm limit using redis.incr
    """

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour

    def __init__(
        self, router_cache: DualCache, model_list: list, routing_args: dict = {}
    ):
        self.router_cache = router_cache
        self.model_list = model_list
        self.routing_args = RoutingArgs(**routing_args)
        BaseRoutingStrategy.__init__(
            self,
            dual_cache=router_cache,
            should_batch_redis_writes=True,
            default_sync_interval=0.1,
        )

    def pre_call_check(self, deployment: Dict) -> Optional[Dict]:
        """
        Pre-call check + update model rpm

        Returns - deployment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        try:
            # ------------
            # Setup values
            # ------------

            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            model_id = deployment.get("model_info", {}).get("id")
            deployment_name = deployment.get("litellm_params", {}).get("model")
            rpm_key = f"{model_id}:{deployment_name}:rpm:{current_minute}"

            local_result = self.router_cache.get_cache(
                key=rpm_key, local_only=True
            )  # check local result first

            deployment_rpm = None
            if deployment_rpm is None:
                deployment_rpm = deployment.get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("litellm_params", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("model_info", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = float("inf")

            if local_result is not None and local_result >= deployment_rpm:
                raise litellm.RateLimitError(
                    message="Deployment over defined rpm limit={}. current usage={}".format(
                        deployment_rpm, local_result
                    ),
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content="{} rpm limit={}. current usage={}. id={}, model_group={}. Get the model info by calling 'router.get_model_info(id)".format(
                            RouterErrors.user_defined_ratelimit_error.value,
                            deployment_rpm,
                            local_result,
                            model_id,
                            deployment.get("model_name", ""),
                        ),
                        request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                    ),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks

                result = self.router_cache.increment_cache(
                    key=rpm_key, value=1, ttl=self.routing_args.ttl
                )
                if result is not None and result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message="Deployment over defined rpm limit={}. current usage={}".format(
                            deployment_rpm, result
                        ),
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content="{} rpm limit={}. current usage={}".format(
                                RouterErrors.user_defined_ratelimit_error.value,
                                deployment_rpm,
                                result,
                            ),
                            request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                    )
            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    async def async_pre_call_check(
        self, deployment: Dict, parent_otel_span: Optional[Span]
    ) -> Optional[Dict]:
        """
        Pre-call check + update model rpm
        - Used inside semaphore
        - raise rate limit error if deployment over limit

        Why? solves concurrency issue - https://github.com/BerriAI/litellm/issues/2994

        Returns - deployment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        try:
            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            model_id = deployment.get("model_info", {}).get("id")
            deployment_name = deployment.get("litellm_params", {}).get("model")

            rpm_key = f"{model_id}:{deployment_name}:rpm:{current_minute}"
            local_result = await self.router_cache.async_get_cache(
                key=rpm_key, local_only=True
            )  # check local result first

            deployment_rpm = None
            if deployment_rpm is None:
                deployment_rpm = deployment.get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("litellm_params", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("model_info", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = float("inf")
            if local_result is not None and local_result >= deployment_rpm:
                raise litellm.RateLimitError(
                    message="Deployment over defined rpm limit={}. current usage={}".format(
                        deployment_rpm, local_result
                    ),
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content="{} rpm limit={}. current usage={}".format(
                            RouterErrors.user_defined_ratelimit_error.value,
                            deployment_rpm,
                            local_result,
                        ),
                        headers={"retry-after": str(60)},  # type: ignore
                        request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                    ),
                    num_retries=deployment.get("num_retries"),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks
                result = await self._increment_value_in_current_window(
                    key=rpm_key, value=1, ttl=self.routing_args.ttl
                )
                if result is not None and result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message="Deployment over defined rpm limit={}. current usage={}".format(
                            deployment_rpm, result
                        ),
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content="{} rpm limit={}. current usage={}".format(
                                RouterErrors.user_defined_ratelimit_error.value,
                                deployment_rpm,
                                result,
                            ),
                            headers={"retry-after": str(60)},  # type: ignore
                            request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                        num_retries=deployment.get("num_retries"),
                    )
            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM/RPM usage on success
            """
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group = standard_logging_object.get("model_group")
            model = standard_logging_object["hidden_params"].get("litellm_model_name")
            id = standard_logging_object.get("model_id")
            if model_group is None or id is None or model is None:
                return
            elif isinstance(id, int):
                id = str(id)

            total_tokens = standard_logging_object.get("total_tokens")

            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime(
                "%H-%M"
            )  # use the same timezone regardless of system clock

            tpm_key = f"{id}:{model}:tpm:{current_minute}"
            # ------------
            # Update usage
            # ------------
            # update cache

            ## TPM
            self.router_cache.increment_cache(
                key=tpm_key, value=total_tokens, ttl=self.routing_args.ttl
            )
            ## TPD (daily tokens) - sliding window implementation
            self._update_sliding_window_cache(
                id=id,
                model=model,
                value=total_tokens,
                window_type="tpd",
            )
            ## RPD (daily requests) - sliding window implementation
            self._update_sliding_window_cache(
                id=id,
                model=model,
                value=1,
                window_type="rpd",
            )
            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.lowest_tpm_rpm_v2.py::log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    def _update_sliding_window_cache(
        self,
        id: str,
        model: str,
        value: int,
        window_type: str,  # "tpd" or "rpd"
    ):
        """
        Update sliding window cache for RPD/TPD tracking (sync version).
        Uses a 24-hour sliding window instead of calendar day.
        """
        import time

        from litellm.types.router import RouterCacheEnum

        # Define window size (24 hours in seconds)
        window_size = 24 * 60 * 60
        now_timestamp = int(time.time())

        # Get the appropriate cache keys
        if window_type == "tpd":
            window_key = RouterCacheEnum.TPD_SLIDING_WINDOW.value.format(
                id=id, model=model
            )
            counter_key = RouterCacheEnum.TPD_SLIDING_COUNTER.value.format(
                id=id, model=model
            )
        elif window_type == "rpd":
            window_key = RouterCacheEnum.RPD_SLIDING_WINDOW.value.format(
                id=id, model=model
            )
            counter_key = RouterCacheEnum.RPD_SLIDING_COUNTER.value.format(
                id=id, model=model
            )
        else:
            raise ValueError(f"Invalid window_type: {window_type}")

        # Get current window start time
        window_start = self.router_cache.get_cache(key=window_key)

        if window_start is None or (now_timestamp - int(window_start)) >= window_size:
            # Reset window and counter - start new 24-hour window
            self.router_cache.set_cache(
                key=window_key,
                value=now_timestamp,
                ttl=window_size,
            )
            self.router_cache.set_cache(
                key=counter_key,
                value=value,
                ttl=window_size,
            )
        else:
            # Increment counter within existing window
            self.router_cache.increment_cache(
                key=counter_key,
                value=value,
                ttl=window_size,
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM usage on success
            """
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group = standard_logging_object.get("model_group")
            model = standard_logging_object["hidden_params"]["litellm_model_name"]
            id = standard_logging_object.get("model_id")
            if model_group is None or id is None:
                return
            elif isinstance(id, int):
                id = str(id)
            total_tokens = standard_logging_object.get("total_tokens")
            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime(
                "%H-%M"
            )  # use the same timezone regardless of system clock

            tpm_key = f"{id}:{model}:tpm:{current_minute}"
            # ------------
            # Update usage
            # ------------
            # update cache
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            ## TPM
            await self.router_cache.async_increment_cache(
                key=tpm_key,
                value=total_tokens,
                ttl=self.routing_args.ttl,
                parent_otel_span=parent_otel_span,
            )
            ## TPD (daily tokens) - sliding window implementation
            await self._async_update_sliding_window_cache(
                id=id,
                model=model,
                value=total_tokens,
                window_type="tpd",
                parent_otel_span=parent_otel_span,
            )
            ## RPD (daily requests) - sliding window implementation
            await self._async_update_sliding_window_cache(
                id=id,
                model=model,
                value=1,
                window_type="rpd",
                parent_otel_span=parent_otel_span,
            )

            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.lowest_tpm_rpm_v2.py::async_log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    async def _async_update_sliding_window_cache(
        self,
        id: str,
        model: str,
        value: int,
        window_type: str,  # "tpd" or "rpd"
        parent_otel_span=None,
    ):
        """
        Update sliding window cache for RPD/TPD tracking.
        Uses a 24-hour sliding window instead of calendar day.
        """
        import time

        from litellm.types.router import RouterCacheEnum

        # Define window size (24 hours in seconds)
        window_size = 24 * 60 * 60
        now_timestamp = int(time.time())

        # Get the appropriate cache keys
        if window_type == "tpd":
            window_key = RouterCacheEnum.TPD_SLIDING_WINDOW.value.format(
                id=id, model=model
            )
            counter_key = RouterCacheEnum.TPD_SLIDING_COUNTER.value.format(
                id=id, model=model
            )
        elif window_type == "rpd":
            window_key = RouterCacheEnum.RPD_SLIDING_WINDOW.value.format(
                id=id, model=model
            )
            counter_key = RouterCacheEnum.RPD_SLIDING_COUNTER.value.format(
                id=id, model=model
            )
        else:
            raise ValueError(f"Invalid window_type: {window_type}")

        # Get current window start time
        window_start = await self.router_cache.async_get_cache(
            key=window_key,
            parent_otel_span=parent_otel_span,
        )

        if window_start is None or (now_timestamp - int(window_start)) >= window_size:
            # Reset window and counter - start new 24-hour window
            await self.router_cache.async_set_cache(
                key=window_key,
                value=now_timestamp,
                ttl=window_size,
                parent_otel_span=parent_otel_span,
            )
            await self.router_cache.async_set_cache(
                key=counter_key,
                value=value,
                ttl=window_size,
                parent_otel_span=parent_otel_span,
            )
        else:
            # Increment counter within existing window
            await self.router_cache.async_increment_cache(
                key=counter_key,
                value=value,
                ttl=window_size,
                parent_otel_span=parent_otel_span,
            )

    def _get_deployment_limit(self, deployment: Dict, limit_type: str) -> float:
        """Get deployment limit (tpm, rpm, tpd, rpd) from deployment configuration."""
        limit = deployment.get(limit_type)
        if limit is None:
            limit = deployment.get("litellm_params", {}).get(limit_type)
        if limit is None:
            limit = deployment.get("model_info", {}).get(limit_type)
        return limit if limit is not None else float("inf")

    def _find_deployment_by_id(
        self, healthy_deployments: List[Dict], item_id: str
    ) -> Optional[Dict]:
        """Find deployment in healthy_deployments by item_id."""
        for m in healthy_deployments:
            if item_id == m["model_info"]["id"]:
                return m
        return None

    def _check_deployment_limits(
        self,
        item: str,
        item_tpm: int,
        input_tokens: int,
        deployment: Dict,
        rpm_dict: Dict,
        tpd_dict: Optional[Dict],
        rpd_dict: Optional[Dict],
    ) -> bool:
        """Check if deployment exceeds any limits. Returns True if within limits."""
        deployment_tpm = self._get_deployment_limit(deployment, "tpm")
        deployment_rpm = self._get_deployment_limit(deployment, "rpm")
        deployment_tpd = self._get_deployment_limit(deployment, "tpd")
        deployment_rpd = self._get_deployment_limit(deployment, "rpd")

        # Check TPM limit
        if item_tpm + input_tokens > deployment_tpm:
            return False

        # Check RPM limit
        if (
            rpm_dict is not None
            and item in rpm_dict
            and rpm_dict[item] is not None
            and rpm_dict[item] + 1 >= deployment_rpm
        ):
            return False

        # Check daily token limit
        if (
            tpd_dict is not None
            and item in tpd_dict
            and tpd_dict[item] is not None
            and tpd_dict[item] + input_tokens >= deployment_tpd
        ):
            return False

        # Check daily request limit
        if (
            rpd_dict is not None
            and item in rpd_dict
            and rpd_dict[item] is not None
            and rpd_dict[item] + 1 >= deployment_rpd
        ):
            return False

        return True

    def _return_potential_deployments(
        self,
        healthy_deployments: List[Dict],
        all_deployments: Dict,
        input_tokens: int,
        rpm_dict: Dict,
        tpd_dict: Optional[Dict] = None,
        rpd_dict: Optional[Dict] = None,
    ):
        lowest_tpm = float("inf")
        potential_deployments = []  # if multiple deployments have the same low value

        for item, item_tpm in all_deployments.items():
            # Get the item from model list
            item_id = item.split(":")[0]
            deployment = self._find_deployment_by_id(healthy_deployments, item_id)

            if deployment is None or item_tpm is None:
                continue  # skip to next one

            # Check if deployment is within all limits
            if not self._check_deployment_limits(
                item, item_tpm, input_tokens, deployment, rpm_dict, tpd_dict, rpd_dict
            ):
                continue

            # Track deployments with lowest TPM
            if item_tpm == lowest_tpm:
                potential_deployments.append(deployment)
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                potential_deployments = [deployment]

        return potential_deployments

    def _common_checks_available_deployment(  # noqa: PLR0915
        self,
        model_group: str,
        healthy_deployments: list,
        tpm_keys: list,
        tpm_values: Optional[list],
        rpm_keys: list,
        rpm_values: Optional[list],
        tpd_keys: Optional[list] = None,
        tpd_values: Optional[list] = None,
        rpd_keys: Optional[list] = None,
        rpd_values: Optional[list] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ) -> Optional[dict]:
        """
        Common checks for get available deployment, across sync + async implementations
        """

        if tpm_values is None or rpm_values is None:
            return None

        tpm_dict = {}  # {model_id: 1, ..}
        for idx, key in enumerate(tpm_keys):
            tpm_dict[tpm_keys[idx].split(":")[0]] = tpm_values[idx]

        rpm_dict = {}  # {model_id: 1, ..}
        for idx, key in enumerate(rpm_keys):
            rpm_dict[rpm_keys[idx].split(":")[0]] = rpm_values[idx]

        # Build daily tracking dictionaries
        tpd_dict = {}  # {model_id: 1, ..}
        if tpd_keys is not None and tpd_values is not None:
            for idx, key in enumerate(tpd_keys):
                # Parse sliding window key format: "global_router:{id}:{model}:tpd_sliding_counter"
                key_parts = tpd_keys[idx].split(":")
                if len(key_parts) >= 3:
                    model_id = key_parts[1]  # Extract {id} from the key
                    tpd_dict[model_id] = tpd_values[idx]

        rpd_dict = {}  # {model_id: 1, ..}
        if rpd_keys is not None and rpd_values is not None:
            for idx, key in enumerate(rpd_keys):
                # Parse sliding window key format: "global_router:{id}:{model}:rpd_sliding_counter"
                key_parts = rpd_keys[idx].split(":")
                if len(key_parts) >= 3:
                    model_id = key_parts[1]  # Extract {id} from the key
                    rpd_dict[model_id] = rpd_values[idx]

        try:
            input_tokens = token_counter(messages=messages, text=input)
        except Exception:
            input_tokens = 0
        verbose_router_logger.debug(f"input_tokens={input_tokens}")
        # -----------------------
        # Find lowest used model
        # ----------------------

        if tpm_dict is None:  # base case - none of the deployments have been used
            # initialize a tpm dict with {model_id: 0}
            tpm_dict = {}
            for deployment in healthy_deployments:
                tpm_dict[deployment["model_info"]["id"]] = 0
        else:
            for d in healthy_deployments:
                ## if healthy deployment not yet used
                tpm_key = d["model_info"]["id"]
                if tpm_key not in tpm_dict or tpm_dict[tpm_key] is None:
                    tpm_dict[tpm_key] = 0

        # Initialize daily dictionaries for unused deployments
        if tpd_dict is None:
            tpd_dict = {}
            for deployment in healthy_deployments:
                tpd_dict[deployment["model_info"]["id"]] = 0
        else:
            for d in healthy_deployments:
                tpd_key = d["model_info"]["id"]
                if tpd_key not in tpd_dict or tpd_dict[tpd_key] is None:
                    tpd_dict[tpd_key] = 0

        if rpd_dict is None:
            rpd_dict = {}
            for deployment in healthy_deployments:
                rpd_dict[deployment["model_info"]["id"]] = 0
        else:
            for d in healthy_deployments:
                rpd_key = d["model_info"]["id"]
                if rpd_key not in rpd_dict or rpd_dict[rpd_key] is None:
                    rpd_dict[rpd_key] = 0

        all_deployments = tpm_dict
        potential_deployments = self._return_potential_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
            input_tokens=input_tokens,
            rpm_dict=rpm_dict,
            tpd_dict=tpd_dict,
            rpd_dict=rpd_dict,
        )
        print_verbose("returning picked lowest tpm/rpm deployment.")

        if len(potential_deployments) > 0:
            return random.choice(potential_deployments)
        else:
            return None

    def _create_cache_keys(
        self, healthy_deployments: list, current_minute: str, current_day: str
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Create cache keys for TPM, RPM, TPD, and RPD tracking."""
        from litellm.types.router import RouterCacheEnum

        tpm_keys = []
        rpm_keys = []
        tpd_keys = []
        rpd_keys = []

        for m in healthy_deployments:
            if isinstance(m, dict):
                id = m.get("model_info", {}).get("id")
                deployment_name = m.get("litellm_params", {}).get("model")
                tpm_key = "{}:{}:tpm:{}".format(id, deployment_name, current_minute)
                rpm_key = "{}:{}:rpm:{}".format(id, deployment_name, current_minute)

                # Use sliding window keys for daily tracking
                tpd_key = RouterCacheEnum.TPD_SLIDING_COUNTER.value.format(
                    id=id, model=deployment_name
                )
                rpd_key = RouterCacheEnum.RPD_SLIDING_COUNTER.value.format(
                    id=id, model=deployment_name
                )

                tpm_keys.append(tpm_key)
                rpm_keys.append(rpm_key)
                tpd_keys.append(tpd_key)
                rpd_keys.append(rpd_key)

        return tpm_keys, rpm_keys, tpd_keys, rpd_keys

    def _split_cache_values(
        self, combined_values: List, tpm_keys: List, rpm_keys: List, tpd_keys: List
    ) -> Tuple[List, List, List, List]:
        """Split combined cache values into separate TPM, RPM, TPD, and RPD lists."""
        if combined_values is None:
            return None, None, None, None

        tpm_values = combined_values[: len(tpm_keys)]
        rpm_values = combined_values[len(tpm_keys) : len(tpm_keys) + len(rpm_keys)]
        tpd_values = combined_values[
            len(tpm_keys)
            + len(rpm_keys) : len(tpm_keys)
            + len(rpm_keys)
            + len(tpd_keys)
        ]
        rpd_values = combined_values[len(tpm_keys) + len(rpm_keys) + len(tpd_keys) :]

        return tpm_values, rpm_values, tpd_values, rpd_values

    def _build_deployment_dict_for_error(
        self, healthy_deployments: list, tpm_values: List, rpm_values: List
    ) -> Dict:
        """Build deployment dictionary for error reporting."""
        deployment_dict = {}
        for index, _deployment in enumerate(healthy_deployments):
            if isinstance(_deployment, dict):
                id = _deployment.get("model_info", {}).get("id")
                deployment_tpm = self._get_deployment_limit(_deployment, "tpm")
                deployment_rpm = self._get_deployment_limit(_deployment, "rpm")

                current_tpm = tpm_values[index] if tpm_values else 0
                current_rpm = rpm_values[index] if rpm_values else 0

                deployment_dict[id] = {
                    "current_tpm": current_tpm,
                    "tpm_limit": deployment_tpm,
                    "current_rpm": current_rpm,
                    "rpm_limit": deployment_rpm,
                }
        return deployment_dict

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ):
        """
        Async implementation of get deployments.

        Reduces time to retrieve the tpm/rpm values from cache
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            f"get_available_deployments - Usage Based. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        current_day = dt.strftime("%Y-%m-%d")

        tpm_keys, rpm_keys, tpd_keys, rpd_keys = self._create_cache_keys(
            healthy_deployments, current_minute, current_day
        )

        combined_tpm_rpm_keys = tpm_keys + rpm_keys + tpd_keys + rpd_keys
        combined_tpm_rpm_values = await self.router_cache.async_batch_get_cache(
            keys=combined_tpm_rpm_keys
        )

        tpm_values, rpm_values, tpd_values, rpd_values = self._split_cache_values(
            combined_tpm_rpm_values, tpm_keys, rpm_keys, tpd_keys
        )

        deployment = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_keys=tpm_keys,
            tpm_values=tpm_values,
            rpm_keys=rpm_keys,
            rpm_values=rpm_values,
            tpd_keys=tpd_keys,
            tpd_values=tpd_values,
            rpd_keys=rpd_keys,
            rpd_values=rpd_values,
            messages=messages,
            input=input,
        )

        if deployment is not None:
            return deployment

        # Build error information and raise exception
        deployment_dict = self._build_deployment_dict_for_error(
            healthy_deployments, tpm_values, rpm_values
        )
        raise litellm.RateLimitError(
            message=f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}",
            llm_provider="",
            model=model_group,
            response=httpx.Response(
                status_code=429,
                content="",
                headers={"retry-after": str(60)},  # type: ignore
                request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
            ),
        )

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        parent_otel_span: Optional[Span] = None,
    ):
        """
        Returns a deployment with the lowest TPM/RPM usage.
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            f"get_available_deployments - Usage Based. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        current_day = dt.strftime("%Y-%m-%d")

        tpm_keys, rpm_keys, tpd_keys, rpd_keys = self._create_cache_keys(
            healthy_deployments, current_minute, current_day
        )

        tpm_values = self.router_cache.batch_get_cache(
            keys=tpm_keys, parent_otel_span=parent_otel_span
        )
        rpm_values = self.router_cache.batch_get_cache(
            keys=rpm_keys, parent_otel_span=parent_otel_span
        )
        tpd_values = self.router_cache.batch_get_cache(
            keys=tpd_keys, parent_otel_span=parent_otel_span
        )
        rpd_values = self.router_cache.batch_get_cache(
            keys=rpd_keys, parent_otel_span=parent_otel_span
        )

        deployment = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_keys=tpm_keys,
            tpm_values=tpm_values,
            rpm_keys=rpm_keys,
            rpm_values=rpm_values,
            tpd_keys=tpd_keys,
            tpd_values=tpd_values,
            rpd_keys=rpd_keys,
            rpd_values=rpd_values,
            messages=messages,
            input=input,
        )

        if deployment is not None:
            return deployment

        # Build error information and raise exception
        deployment_dict = self._build_deployment_dict_for_error(
            healthy_deployments, tpm_values, rpm_values
        )
        raise ValueError(
            f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}"
        )
