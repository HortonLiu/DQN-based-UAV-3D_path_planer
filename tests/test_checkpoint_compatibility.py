"""Checkpoint loading must work on both the user's Torch 1.11 and newer Torch."""

import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'BaseClass', ROOT / 'Trainer'):
    sys.path.insert(0, str(directory))

import simulator as simulator_module

trainer_module = importlib.import_module('SAC_Trainer')
factory_module = importlib.import_module('FactoryClass.TrainerFactory')


class CheckpointLoadCompatibilityTests(unittest.TestCase):
    def test_legacy_torch_load_does_not_receive_weights_only_in_pickle_arguments(self):
        calls = []
        checkpoint = {'model': 'legacy weights', 'epoch': 7}

        def legacy_load(path, map_location=None, **pickle_load_args):
            # Torch 1.11 forwards these kwargs into Unpickler instead of handling
            # weights_only itself; passing it would reproduce the user's error.
            if 'weights_only' in pickle_load_args:
                raise TypeError("'weights_only' is an invalid keyword argument for Unpickler()")
            calls.append((path, map_location, pickle_load_args))
            return checkpoint

        with patch.object(trainer_module.torch, 'load', legacy_load):
            restored = trainer_module._load_checkpoint('local_model.pth', 'cpu')
        self.assertIs(restored, checkpoint)
        self.assertEqual(calls, [('local_model.pth', 'cpu', {})])

    def test_modern_torch_load_keeps_restricted_weights_only_loading(self):
        calls = []
        checkpoint = {'model': 'modern weights', 'epoch': 11}

        def modern_load(path, map_location=None, *, weights_only=False, **kwargs):
            calls.append((path, map_location, weights_only, kwargs))
            return checkpoint

        with patch.object(trainer_module.torch, 'load', modern_load):
            restored = trainer_module._load_checkpoint('local_model.pth', 'cpu')
        self.assertIs(restored, checkpoint)
        self.assertEqual(calls, [('local_model.pth', 'cpu', True, {})])

    def test_corrupt_or_unsupported_checkpoint_is_not_retried_unrestricted(self):
        calls = []
        failure = RuntimeError('checkpoint archive is incomplete')

        def modern_load(path, map_location=None, *, weights_only=False, **kwargs):
            calls.append(weights_only)
            raise failure

        with patch.object(trainer_module.torch, 'load', modern_load):
            with self.assertRaises(RuntimeError) as raised:
                trainer_module._load_checkpoint('broken_model.pth', 'cpu')
        self.assertIs(raised.exception, failure)
        self.assertEqual(calls, [True])


class TrainerInitializationFailureTests(unittest.TestCase):
    def test_factory_preserves_original_initialization_error_instead_of_returning_none(self):
        failure = TypeError('model checkpoint could not be loaded')

        class BrokenTrainer:
            def __init__(self, parameters):
                raise failure

        module = types.SimpleNamespace(BrokenTrainer=BrokenTrainer)
        with patch.object(factory_module.importlib, 'import_module', return_value=module):
            with self.assertRaises(RuntimeError) as raised:
                factory_module.TrainerFactory().Create_Trainer({'Trainer_Type': 'BrokenTrainer'})
        self.assertIs(raised.exception.__cause__, failure)
        self.assertIn('BrokenTrainer', str(raised.exception))

    def test_simulator_rejects_missing_trainer_before_records_and_startup_messages(self):
        agent = types.SimpleNamespace(name='UAV_test', Trainer=None)
        environment = types.SimpleNamespace(Agents=[agent])
        with patch.object(simulator_module.EnvFactory, 'Create_Env', return_value=environment):
            with patch.object(simulator_module.simulator, 'Init_Record_Mod') as initialize_records:
                with self.assertRaises(RuntimeError) as raised:
                    simulator_module.simulator(episodes=1, no_train=True)
        self.assertIn('trainer', str(raised.exception).lower())
        initialize_records.assert_not_called()


if __name__ == '__main__':
    unittest.main()
