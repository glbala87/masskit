"""Tests for plugin architecture."""

import pytest

from masskit.plugins import (
    PluginRegistry,
    PluginInfo,
    ProcessingPipeline,
    register_as,
)


@pytest.fixture(autouse=True)
def reset_registry():
    PluginRegistry.reset()
    yield
    PluginRegistry.reset()


class TestPluginRegistry:
    def test_register_algorithm(self):
        registry = PluginRegistry()
        registry.register_algorithm("test_algo", lambda x: x * 2)
        assert "test_algo" in registry.list_algorithms()

    def test_get_algorithm(self):
        registry = PluginRegistry()
        registry.register_algorithm("double", lambda x: x * 2)
        func = registry.get_algorithm("double")
        assert func(5) == 10

    def test_missing_algorithm(self):
        registry = PluginRegistry()
        with pytest.raises(KeyError):
            registry.get_algorithm("nonexistent")

    def test_register_reader(self):
        registry = PluginRegistry()
        registry.register_reader("custom_format", lambda f: None, extensions=[".cust"])
        assert "custom_format" in registry.list_readers()

    def test_register_writer(self):
        registry = PluginRegistry()
        registry.register_writer("custom_out", lambda data, f: None)
        assert "custom_out" in registry.list_writers()

    def test_register_processor(self):
        registry = PluginRegistry()
        registry.register_processor("normalize", lambda x: x / max(x))
        assert "normalize" in registry.list_processors()

    def test_list_plugins(self):
        registry = PluginRegistry()
        registry.register_algorithm("a1", lambda: None)
        registry.register_reader("r1", lambda: None)
        plugins = registry.list_plugins()
        assert "algorithms" in plugins
        assert "a1" in plugins["algorithms"]

    def test_singleton(self):
        r1 = PluginRegistry.instance()
        r2 = PluginRegistry.instance()
        assert r1 is r2


class TestHooks:
    def test_register_and_trigger(self):
        registry = PluginRegistry()
        called = []
        registry.register_hook("pre_process", lambda **kw: called.append("pre"))
        registry.trigger_hook("pre_process")
        assert called == ["pre"]

    def test_hook_error_handling(self):
        registry = PluginRegistry()
        registry.register_hook("pre_process", lambda **kw: 1/0)
        # Should not raise
        registry.trigger_hook("pre_process")


class TestPipeline:
    def test_basic_pipeline(self):
        registry = PluginRegistry()
        registry.register_processor("add_one", lambda x: x + 1)
        registry.register_processor("double", lambda x: x * 2)

        pipeline = ProcessingPipeline(registry)
        pipeline.add_step("add_one")
        pipeline.add_step("double")

        result = pipeline.run(5)
        assert result == 12  # (5 + 1) * 2

    def test_pipeline_chaining(self):
        registry = PluginRegistry()
        registry.register_processor("inc", lambda x: x + 1)

        pipeline = ProcessingPipeline(registry)
        pipeline.add_step("inc").add_step("inc").add_step("inc")
        assert len(pipeline.steps) == 3
        assert pipeline.run(0) == 3

    def test_missing_step(self):
        registry = PluginRegistry()
        pipeline = ProcessingPipeline(registry)
        pipeline.add_step("nonexistent")
        with pytest.raises(KeyError):
            pipeline.run(0)


class TestDecorator:
    def test_register_as(self):
        @register_as("algorithm", "my_algo")
        def my_algo(x):
            return x ** 2

        registry = PluginRegistry.instance()
        assert "my_algo" in registry.list_algorithms()
        assert registry.get_algorithm("my_algo")(4) == 16
