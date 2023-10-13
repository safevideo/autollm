from llama_index import ServiceContext as LlamaServiceContext

from autollm.utils.llm_utils import initialize_token_counting


class ServiceContext(LlamaServiceContext):
    """
    ServiceContext extends the functionality of LlamaIndex's ServiceContext to include token counting.

    This class is initialized with an optional token_counter and all the parameters required for LlamaIndex's
    ServiceContext.
    """

    def __init__(self, token_counter, *args, **kwargs):
        self._token_counter = token_counter
        super().__init__(*args, **kwargs)

    @property
    def token_counter(self):
        """Property to access the token_counter."""
        return self._token_counter

    @classmethod
    def from_defaults(cls, enable_cost_logging: bool = False, *args, **kwargs) -> "ServiceContext":
        """
        Create a ServiceContext from defaults.

        Parameters:
            *args, **kwargs: Arguments for the parent ServiceContext class.

        Returns:
            ServiceContext: Initialized service context with token_counter attached.
        """
        # Initialize token counting if enable_cost_logging is True
        token_counter = None
        if enable_cost_logging:
            # from your llm_utils module
            token_counter, callback_manager = initialize_token_counting()
            if callback_manager:
                kwargs['callback_manager'] = callback_manager

        # Initialize the parent ServiceContext with the given or default parameters
        service_context = cls(token_counter=token_counter, *args, **kwargs)

        return service_context


class AutoServiceContext:

    @classmethod
    def from_defaults(cls, enable_cost_logging: bool = False, *args, **kwargs) -> ServiceContext:
        """
        Create an AutoServiceContext from defaults. If enable_cost_logging is True, initializes the token
        counter.

        Parameters:
            enable_cost_logging (bool): Flag to enable cost logging.
            *args, **kwargs: Arguments for the parent ServiceContext class.

        Returns:
            ServiceContext: Initialized service context with token_counter attached.
        """

        return ServiceContext.from_defaults(enable_cost_logging=enable_cost_logging, *args, **kwargs)


# The idea is to use it like this:
# service_context = AutoServiceContext.from_defaults(enable_cost_logging=True, llm="gpt-3.5-turbo")
