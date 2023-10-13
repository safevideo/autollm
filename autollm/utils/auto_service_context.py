from typing import Optional, Type

from llama_index import ServiceContext
from llama_index import TokenCounter, CallbackManager

from .llm_utils import initialize_token_counting


class AutoServiceContext(ServiceContext):
    """
    AutoServiceContext extends the functionality of LlamaIndex's ServiceContext to include token counting.
    This class is initialized with an optional token_counter and all the parameters required for ServiceContext.
    """
    def __init__(self, *args, **kwargs):
        self._token_counter = None
        super().__init__(*args, **kwargs)

    @property
    def token_counter(self):
        """Property to access the token_counter."""
        return self._token_counter

    @classmethod
    def from_defaults(cls, enable_cost_logging: bool = False, *args, **kwargs) -> ServiceContext:
        """
        Create an AutoServiceContext from defaults.
        If enable_cost_logging is True, initializes the token counter.
        
        Parameters:
            enable_cost_logging (bool): Flag to enable cost logging.
            *args, **kwargs: Arguments for the parent ServiceContext class.

        Returns:
            ServiceContext: Initialized service context with token_counter attached.
        """
        # Initialize token counting if enable_cost_logging is True
        if enable_cost_logging:
            token_counter, callback_manager = initialize_token_counting()  # from your llm_utils module
            if callback_manager:
                kwargs["callback_manager"] = callback_manager

        # Initialize the parent ServiceContext with the given or default parameters
        auto_service_context = super(AutoServiceContext, cls).from_defaults(*args, **kwargs)
        
        # Attach the token_counter to this AutoServiceContext
        auto_service_context._token_counter = token_counter

        return auto_service_context

# The idea is to use it like this:
# service_context = AutoServiceContext.from_defaults(enable_cost_logging=True, llm="gpt-3.5-turbo")
