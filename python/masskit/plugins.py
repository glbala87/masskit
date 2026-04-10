"""
Plugin architecture for MassKit.

Allows users to register custom algorithms, file readers/writers,
and processing steps without modifying the core library.
"""

from typing import Optional, List, Dict, Any, Callable, Type
from dataclasses import dataclass, field
import importlib
import pkgutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""
    name: str = ""
    version: str = "0.0.0"
    author: str = ""
    description: str = ""
    category: str = ""  # 'algorithm', 'reader', 'writer', 'processor'
    entry_point: str = ""  # module:class or module:function


class PluginRegistry:
    """
    Central registry for MassKit plugins.

    Plugins can be registered programmatically or discovered from
    installed packages via entry points.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_algorithm("my_peak_picker", my_function)
        >>> picker = registry.get_algorithm("my_peak_picker")
        >>> peaks = picker(spectrum)
    """

    _instance: Optional["PluginRegistry"] = None

    def __init__(self):
        self._algorithms: Dict[str, Callable] = {}
        self._readers: Dict[str, Callable] = {}
        self._writers: Dict[str, Callable] = {}
        self._processors: Dict[str, Callable] = {}
        self._plugin_info: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = {}

    @classmethod
    def instance(cls) -> "PluginRegistry":
        """Get the global plugin registry singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global registry (mainly for testing)."""
        cls._instance = None

    # --- Algorithm registration ---

    def register_algorithm(
        self,
        name: str,
        func: Callable,
        info: Optional[PluginInfo] = None,
    ) -> None:
        """Register a custom algorithm."""
        self._algorithms[name] = func
        if info:
            info.category = "algorithm"
            self._plugin_info[f"algorithm:{name}"] = info
        logger.debug(f"Registered algorithm: {name}")

    def get_algorithm(self, name: str) -> Callable:
        """Get a registered algorithm by name."""
        if name not in self._algorithms:
            raise KeyError(f"Algorithm '{name}' not registered. "
                           f"Available: {list(self._algorithms.keys())}")
        return self._algorithms[name]

    def list_algorithms(self) -> List[str]:
        return list(self._algorithms.keys())

    # --- Reader registration ---

    def register_reader(
        self,
        format_name: str,
        func: Callable,
        extensions: Optional[List[str]] = None,
        info: Optional[PluginInfo] = None,
    ) -> None:
        """
        Register a custom file reader.

        Args:
            format_name: Format identifier (e.g., 'custom_mzml')
            func: Reader function(filepath) -> MSExperiment or similar
            extensions: File extensions this reader handles
            info: Plugin metadata
        """
        self._readers[format_name] = func
        if extensions:
            for ext in extensions:
                self._readers[f"ext:{ext.lstrip('.')}"] = func
        if info:
            info.category = "reader"
            self._plugin_info[f"reader:{format_name}"] = info

    def get_reader(self, format_name: str) -> Callable:
        """Get a registered reader by format name or file extension."""
        if format_name not in self._readers:
            ext_key = f"ext:{format_name.lstrip('.')}"
            if ext_key in self._readers:
                return self._readers[ext_key]
            raise KeyError(f"Reader '{format_name}' not registered. "
                           f"Available: {list(self._readers.keys())}")
        return self._readers[format_name]

    def list_readers(self) -> List[str]:
        return [k for k in self._readers.keys() if not k.startswith("ext:")]

    # --- Writer registration ---

    def register_writer(
        self,
        format_name: str,
        func: Callable,
        info: Optional[PluginInfo] = None,
    ) -> None:
        """Register a custom file writer."""
        self._writers[format_name] = func
        if info:
            info.category = "writer"
            self._plugin_info[f"writer:{format_name}"] = info

    def get_writer(self, format_name: str) -> Callable:
        if format_name not in self._writers:
            raise KeyError(f"Writer '{format_name}' not registered.")
        return self._writers[format_name]

    def list_writers(self) -> List[str]:
        return list(self._writers.keys())

    # --- Processor registration ---

    def register_processor(
        self,
        name: str,
        func: Callable,
        info: Optional[PluginInfo] = None,
    ) -> None:
        """
        Register a custom spectrum/data processor.

        Processors are functions that take data and return transformed data.
        """
        self._processors[name] = func
        if info:
            info.category = "processor"
            self._plugin_info[f"processor:{name}"] = info

    def get_processor(self, name: str) -> Callable:
        if name not in self._processors:
            raise KeyError(f"Processor '{name}' not registered.")
        return self._processors[name]

    def list_processors(self) -> List[str]:
        return list(self._processors.keys())

    # --- Hooks ---

    def register_hook(self, event: str, callback: Callable) -> None:
        """
        Register a callback for an event hook.

        Events: 'pre_load', 'post_load', 'pre_process', 'post_process',
                'pre_save', 'post_save'
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def trigger_hook(self, event: str, **kwargs) -> None:
        """Trigger all callbacks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                logger.warning(f"Hook '{event}' callback failed: {e}")

    # --- Plugin info ---

    def get_plugin_info(self, key: str) -> Optional[PluginInfo]:
        return self._plugin_info.get(key)

    def list_plugins(self) -> Dict[str, List[str]]:
        """List all registered plugins by category."""
        return {
            "algorithms": self.list_algorithms(),
            "readers": self.list_readers(),
            "writers": self.list_writers(),
            "processors": self.list_processors(),
        }

    # --- Discovery ---

    def discover_plugins(self, namespace: str = "masskit_plugins") -> int:
        """
        Discover and load plugins from installed packages.

        Searches for packages in the given namespace (default: masskit_plugins).
        Plugin packages should have a `register(registry)` function.

        Args:
            namespace: Package namespace to search

        Returns:
            Number of plugins discovered
        """
        count = 0
        try:
            ns_pkg = importlib.import_module(namespace)
        except ImportError:
            return 0

        for importer, modname, ispkg in pkgutil.iter_modules(
            ns_pkg.__path__, namespace + "."
        ):
            try:
                module = importlib.import_module(modname)
                if hasattr(module, "register"):
                    module.register(self)
                    count += 1
                    logger.info(f"Loaded plugin: {modname}")
            except Exception as e:
                logger.warning(f"Failed to load plugin {modname}: {e}")

        return count

    def load_plugin_module(self, module_path: str) -> None:
        """
        Load a plugin from a Python module path.

        The module should have a `register(registry)` function.

        Args:
            module_path: Dotted module path (e.g., 'mypackage.my_plugin')
        """
        module = importlib.import_module(module_path)
        if hasattr(module, "register"):
            module.register(self)
        else:
            raise AttributeError(
                f"Module '{module_path}' has no 'register(registry)' function"
            )


class ProcessingPipeline:
    """
    Configurable processing pipeline using registered plugins.

    Example:
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.add_step("smooth", window_size=5)
        >>> pipeline.add_step("pick_peaks", snr=3.0)
        >>> pipeline.add_step("my_custom_filter")  # registered plugin
        >>> results = pipeline.run(spectrum)
    """

    def __init__(self, registry: Optional[PluginRegistry] = None):
        self._registry = registry or PluginRegistry.instance()
        self._steps: List[Dict[str, Any]] = []

    def add_step(self, name: str, **kwargs) -> "ProcessingPipeline":
        """Add a processing step. Returns self for chaining."""
        self._steps.append({"name": name, "kwargs": kwargs})
        return self

    def remove_step(self, index: int) -> None:
        """Remove a step by index."""
        self._steps.pop(index)

    def clear(self) -> None:
        """Remove all steps."""
        self._steps.clear()

    @property
    def steps(self) -> List[str]:
        return [s["name"] for s in self._steps]

    def run(self, data: Any) -> Any:
        """
        Run the pipeline on input data.

        Each step's output becomes the next step's input.
        """
        self._registry.trigger_hook("pre_process", data=data)

        result = data
        for step in self._steps:
            name = step["name"]
            kwargs = step["kwargs"]

            # Try processors first, then algorithms
            try:
                func = self._registry.get_processor(name)
            except KeyError:
                try:
                    func = self._registry.get_algorithm(name)
                except KeyError:
                    raise KeyError(
                        f"Step '{name}' not found in processors or algorithms"
                    )

            result = func(result, **kwargs)

        self._registry.trigger_hook("post_process", data=result)
        return result

    def __repr__(self) -> str:
        steps_str = " -> ".join(self.steps)
        return f"ProcessingPipeline({steps_str})"


# Convenience decorator for plugin registration
def register_as(category: str, name: str, **info_kwargs):
    """
    Decorator to register a function as a plugin.

    Example:
        >>> @register_as("algorithm", "my_peak_picker", version="1.0")
        ... def my_peak_picker(spectrum, threshold=100):
        ...     ...
    """
    def decorator(func):
        registry = PluginRegistry.instance()
        info = PluginInfo(name=name, **info_kwargs) if info_kwargs else None

        if category == "algorithm":
            registry.register_algorithm(name, func, info)
        elif category == "reader":
            registry.register_reader(name, func, info=info)
        elif category == "writer":
            registry.register_writer(name, func, info)
        elif category == "processor":
            registry.register_processor(name, func, info)
        else:
            raise ValueError(f"Unknown category: {category}")

        return func
    return decorator
