"""Regression checks for truthful CSV output and isolated corrected-run weights."""
import csv
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import simulator as simulator_module
from BaseClass.CalMod import XML2Dict
from FactoryClass.TrainerFactory import TrainerFactory


class NullProgress:
    def __init__(self, **kwargs):
        self.completed = 0
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def update(self, value): self.completed += value
    def set_postfix(self, **kwargs): pass


class SimulationRecordTests(unittest.TestCase):
    def make_simulator(self, path):
        agent = types.SimpleNamespace(
            Train_time=1.5, Testing_time=0.25, control_hz=480, Control_step=4800,
            sim_time=10.0, Step=9600, Max_Step=1152000, score=8.75,
            path_len=9.8, active_virtual_obstacles=[{}, {}],
            position=types.SimpleNamespace(x=10, y=10, z=0),
            goal=types.SimpleNamespace(x=13, y=14, z=0),
            Trainer=types.SimpleNamespace(epoch=10))
        sim = simulator_module.simulator.__new__(simulator_module.simulator)
        sim.env = types.SimpleNamespace(Agents=[agent], last_render_path='verified.html')
        sim.epoch, sim.executed_time = 3, 7.25
        sim.result_path = str(path)
        sim._result_file = None
        sim.CsvWriter = None
        sim.writer = None
        sim.infos = []
        sim.stat, sim.num_episodes = 1, 3
        sim.max_eps_episode, sim.min_eps = 100, 0.1
        sim.Max_score = float('-inf')
        sim.Init_Record_Mod()
        self.addCleanup(sim.close)
        return sim

    def test_csv_records_real_metrics_and_flushes_while_open(self):
        with tempfile.TemporaryDirectory() as folder:
            sim = self.make_simulator(Path(folder) / 'scores.csv')
            sim.infos.append({'sum_epoch': 10, 'score': 8.75, 'average_score': 8.75,
                              'eps': 0.2, 'success': 1, 'lose': 0, 'meet_threaten': 2,
                              'loss': torch.tensor(0.375), 'step': 9600})
            sim.record_list()
            # A separate reader must see the row immediately, before close().
            with open(sim.result_path, encoding='utf-8', newline='') as reader:
                row = list(csv.DictReader(reader))[0]
            self.assertEqual(float(row[' Score']), 8.75)
            self.assertEqual(float(row['success']), 1)
            self.assertEqual(float(row['failed']), 0)
            self.assertEqual(float(row['meet_threaten']), 2)
            self.assertEqual(float(row['loss']), 0.375)
            self.assertEqual(float(row['step']), 9600)
            self.assertEqual(float(row['control_updates']), 4800)
            self.assertEqual(float(row['simulation_time_s']), 10)
            self.assertEqual(float(row['path_length_m']), 9.8)
            self.assertEqual(float(row['virtual_obstacles']), 2)
            self.assertEqual(float(row['episode_budget_s']), 1200)
            self.assertEqual(float(row['remaining_goal_distance_m']), 5)
            self.assertEqual(row['html_path'], 'verified.html')
            self.assertEqual(sim.infos, [])
            sim.close()

    def test_small_episode_count_is_not_truncated_and_failures_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            sim = self.make_simulator(Path(folder) / 'scores.csv')
            calls = []
            def run(eps):
                calls.append(eps)
                return {'sum_epoch': len(calls), 'score': -2.5, 'average_score': -2.5,
                        'eps': eps, 'success': 0, 'lose': 1, 'meet_threaten': 0,
                        'loss': 0.125, 'step': 9600}
            sim.env.run_eposide = run
            sim.epoch = 0
            with patch.object(simulator_module, 'tqdm', NullProgress):
                sim.StartAndTrain()
            with open(sim.result_path, encoding='utf-8', newline='') as reader:
                rows = list(csv.DictReader(reader))
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(rows), 3)
            self.assertEqual([int(row['Episode']) for row in rows], [1, 2, 3])
            self.assertTrue(all(float(row['failed']) == 1 for row in rows))
            self.assertTrue(all(float(row[' Score']) == -2.5 for row in rows))
            sim.close()


class CorrectedCheckpointTests(unittest.TestCase):
    def make_trainer(self, folder):
        parameters = XML2Dict(str(ROOT / 'config/Trainer.xml'))['Trainer']
        parameters.update(name='checkpoint_test', checkpoint_dir=str(folder))
        trainer = TrainerFactory().Create_Trainer(parameters)
        self.assertIsNotNone(trainer)
        return trainer

    def test_explicit_checkpoint_directory_saves_and_resumes_only_its_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'corrected_weights'
            trainer = self.make_trainer(directory)
            self.assertEqual(Path(trainer.checkpoint_dir), directory)
            trainer.epoch = 7
            expected = next(trainer.actor.parameters()).detach().clone()
            trainer.save()
            self.assertEqual(len(list(directory.glob('*.pth'))), 3)
            restored = self.make_trainer(directory)
            self.assertEqual(restored.epoch, 7)
            torch.testing.assert_close(next(restored.actor.parameters()), expected)

    def test_no_train_does_not_optimize_increment_epoch_or_save(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.make_trainer(Path(folder) / 'input')
            trainer.Is_Train = 0
            before = next(trainer.actor.parameters()).detach().clone()
            result = trainer.update({'states': [[1.0]]})
            self.assertEqual(result, {'sum_epoch': 0, 'loss': 0.0})
            self.assertEqual(trainer.epoch, 0)
            torch.testing.assert_close(next(trainer.actor.parameters()), before)
            output = Path(folder) / 'must_not_be_created'
            trainer.save(output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
