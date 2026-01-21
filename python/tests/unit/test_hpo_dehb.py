import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import ConfigSpace as CS
import pytest
from python.src.pipeline.hpo.de_async import AsyncDifferentialEvolution
from python.src.pipeline.hpo.dehb import DEHB, DifferentialEvolutionHyperband, get_config_space
from python.src.pipeline.hpo.dehb_shb_manager import SynchronousHalvingBracketManager

def dummy_objective(config, fidelity, **kwargs):
    # Simple minimization objective
    if isinstance(config, CS.Configuration):
        config = config.get_dictionary()
    
    fitness = sum(config.values()) / fidelity
    return {"fitness": fitness, "cost": 1.0}

def get_simple_cs():
    cs = CS.ConfigurationSpace()
    cs.add(CS.UniformFloatHyperparameter("x1", 0, 1))
    cs.add(CS.UniformFloatHyperparameter("x2", 0, 1))
    return cs

@pytest.fixture
def basic_dehb_kwargs():
    return {
        "min_budget": 1.0,
        "max_budget": 10.0,
        "eta": 3,
        "mutation_factor": 0.5,
        "crossover_prob": 0.5,
        "strategy": "rand1_bin",
        "output_path": "/tmp/dehb_test",
    }

def test_get_config_space_wcvrp():
    opts = {"problem": "wcvrp", "hop_range": [0.1, 0.9]}
    cs = get_config_space(opts)
    assert "w_lost" in cs
    assert cs["w_lost"].lower == 0.1
    assert cs["w_lost"].upper == 0.9

def test_get_config_space_custom():
    opts = {
        "config_space_params": {
            "p1": (0.0, 10.0),
            "p2": ["a", "b", "c"]
        }
    }
    cs = get_config_space(opts)
    assert "p1" in cs
    assert isinstance(cs["p1"], CS.UniformFloatHyperparameter)
    assert "p2" in cs
    assert isinstance(cs["p2"], CS.CategoricalHyperparameter)

def test_dehb_init(basic_dehb_kwargs):
    cs = get_simple_cs()
    with patch("python.src.pipeline.hpo.dehb.os.makedirs"):
        dehb = DEHB(cs=cs, f=dummy_objective, **basic_dehb_kwargs)
        assert dehb.min_fidelity == 1.0
        assert dehb.max_fidelity == 10.0
        assert dehb.eta == 3
        assert len(dehb.de) > 0
        assert dehb.n_workers == 1

def test_dehb_ask_tell(basic_dehb_kwargs):
    cs = get_simple_cs()
    with patch("python.src.pipeline.hpo.dehb.os.makedirs"):
        dehb = DEHB(cs=cs, f=dummy_objective, **basic_dehb_kwargs)
        
        # Ask for a job
        job = dehb.ask()
        assert "config" in job
        assert "fidelity" in job
        assert "config_id" in job
        
        # Run objective
        result = dummy_objective(job["config"], job["fidelity"])
        
        # Tell result
        dehb.tell(job, result)
        assert dehb._tell_counter == 1
        assert len(dehb.traj) == 1
        assert dehb.inc_score == result["fitness"]

@patch("python.src.pipeline.hpo.dehb.Client")
def test_dehb_distributed_init(mock_client_class, basic_dehb_kwargs):
    mock_client = MagicMock()
    mock_client.scheduler_info.return_value = {"workers": ["w1", "w2"]}
    mock_client_class.return_value = mock_client
    
    cs = get_simple_cs()
    with patch("python.src.pipeline.hpo.dehb.os.makedirs"):
        dehb = DEHB(cs=cs, f=dummy_objective, n_workers=2, **basic_dehb_kwargs)
        # Note: DEHB class sets n_workers=1 in super().__init__ because of how it's wrapped
        # But DifferentialEvolutionHyperband can take n_workers.
        
        # Let's test the base class for distributed
        dehb_base = DifferentialEvolutionHyperband(cs=cs, f=dummy_objective, n_workers=2, **basic_dehb_kwargs)
        assert dehb_base.n_workers == 2

def test_dehb_run_fevals(basic_dehb_kwargs):
    cs = get_simple_cs()
    with patch("python.src.pipeline.hpo.dehb.os.makedirs"), \
         patch("python.src.pipeline.hpo.dehb.DEHB.save"):
        dehb = DEHB(cs=cs, f=dummy_objective, **basic_dehb_kwargs)
        
        traj, runtime, history = dehb.run(fevals=5)
        assert len(traj) == 5
        assert len(runtime) == 5
        assert len(history) == 5

def test_dehb_run_brackets(basic_dehb_kwargs):
    cs = get_simple_cs()
    with patch("python.src.pipeline.hpo.dehb.os.makedirs"), \
         patch("python.src.pipeline.hpo.dehb.DEHB.save"):
        dehb = DEHB(cs=cs, f=dummy_objective, **basic_dehb_kwargs)
        
        # Run for 1 bracket
        traj, runtime, history = dehb.run(brackets=1)
        assert dehb.iteration_counter >= 0
        assert len(traj) > 0

def test_dehb_state_management(basic_dehb_kwargs):
    cs = get_simple_cs()
    with patch("python.src.pipeline.hpo.dehb.os.makedirs"), \
         patch("python.src.pipeline.hpo.dehb.Path.open", unittest.mock.mock_open()):
        dehb = DEHB(cs=cs, f=dummy_objective, **basic_dehb_kwargs)
        
        state = dehb._get_state()
        assert "DE_params" in state
        assert "HB_params" in state
        
        dehb.inc_config = np.array([0.5, 0.5])
        dehb.inc_score = 0.1
        
        with patch("json.dump"):
            dehb._save_incumbent()
            # Verify it doesn't crash

def test_shb_manager():
    n_configs = np.array([9, 3, 1])
    fidelities = np.array([1.0, 3.0, 9.0])
    manager = SynchronousHalvingBracketManager(n_configs, fidelities, bracket_id=0)
    
    assert manager.get_fidelity() == 1.0
    assert manager.get_next_job_fidelity() == 1.0
    
    # Register all jobs for first rung
    for _ in range(9):
        manager.register_job(1.0)
    
    assert manager.current_rung == 1
    assert manager.get_fidelity() == 3.0
    
    # Complete some jobs
    for _ in range(9):
        manager.complete_job(1.0)
    
    assert not manager._is_rung_waiting(0)
    assert manager.is_pending()
    assert manager.get_next_job_fidelity() == 3.0

def test_async_de_mutation():
    cs = get_simple_cs()
    de = AsyncDifferentialEvolution(
        cs=cs, 
        pop_size=10, 
        dimensions=2, 
        mutation_factor=0.5, 
        crossover_prob=0.5,
        strategy="rand1_bin"
    )
    de.population = np.random.uniform(0, 1, (10, 2))
    de.fitness = np.random.uniform(0, 1, 10)
    
    mutant = de.mutation(current=de.population[0], best=de.population[1])
    assert mutant.shape == (2,)
    assert np.all(mutant >= 0) and np.all(mutant <= 1)

def test_dehb_gpu_distribution(basic_dehb_kwargs):
    cs = get_simple_cs()
    with patch("python.src.pipeline.hpo.dehb.os.makedirs"), \
         patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}):
        dehb = DEHB(cs=cs, f=dummy_objective, **basic_dehb_kwargs)
        dehb._distribute_gpus()
        
        assert dehb.available_gpus == [0, 1, 2]
        assert dehb.gpu_usage == {0: 0, 1: 0, 2: 0}
        
        gpu_ids = dehb._get_gpu_id_with_low_load()
        # Should be something like "0,1,2" or "1,0,2" etc with the chosen one first
        assert len(gpu_ids.split(",")) == 3
        # usage should be updated
        assert sum(dehb.gpu_usage.values()) == 1

