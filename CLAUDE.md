# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs) developed by ByteDance Seed team. It implements the HybridFlow framework for reinforcement learning from human feedback (RLHF) with support for various RL algorithms like PPO, GRPO, ReMax, RLOO, GSPO, etc.

## Architecture

The codebase follows a modular architecture with these key components:

1. **Single Controller**: Distributed computing framework with Worker and WorkerGroup abstractions for managing distributed training
2. **Trainer**: High-level training interfaces implementing various RL algorithms (PPO, GRPO, etc.)
3. **Workers**: Specialized components for different roles (actor, critic, reward model, rollout)
4. **Models**: Model implementations and integrations with FSDP, Megatron-LM, vLLM, SGLang
5. **Utils**: Utility functions for distributed training, device management, data handling

Core architectural patterns:
- Hybrid-controller programming model for flexible RL dataflows
- Decoupled computation and data dependencies for integration with existing LLM frameworks
- Flexible device mapping for efficient resource utilization

## Development Environment

### Python Version
Requires Python >= 3.10

### Key Dependencies
- PyTorch for deep learning operations
- Ray for distributed computing
- Transformers for model implementations
- vLLM/SGLang for efficient inference
- FSDP/Megatron-LM for distributed training

## Common Development Commands

### Installation
```bash
# Install with test dependencies
pip install -e .[test]

# Install with vLLM support
pip install -e .[test,vllm]

# Install with SGLang support
pip install -e .[test,sglang]
```

### Code Linting and Formatting
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run pre-commit on staged changes
pre-commit run

# Run pre-commit on all files
pre-commit run --all-files

# Run specific linters
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

### Testing
```bash
# Run CPU unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_base_config_on_cpu.py -v

# Run tests with specific markers
pytest tests/ -m "cpu" -v
```

### Type Checking
```bash
# Run mypy for type checking
mypy verl/
```

### Documentation
```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Build HTML docs
cd docs && make html

# Preview locally
cd docs/_build/html && python -m http.server
```

## Key Configuration Files

- `pyproject.toml`: Build system, linting and type checking configuration
- `setup.py`: Package installation configuration
- `requirements.txt`: Core development dependencies
- Trainer configs in `verl/trainer/config/`: Algorithm and model configurations

## RL Algorithm Implementation Structure

RL algorithms are implemented in:
- Core algorithms: `verl/trainer/ppo/core_algos.py`
- Advantage estimators: Registered functions with `@register_adv_est` decorator
- Policy loss functions: Registered with `@register_policy_loss` decorator
- Trainer implementations: Various files in `verl/trainer/ppo/`

The system supports multiple distributed backends (FSDP, Megatron-LM) and generation engines (vLLM, SGLang).

## New Features

### GSPO Algorithm
Added support for Geometric Sequence Policy Optimization (GSPO) in `core_algos.py`

### Agent Loop Framework
Experimental agent loop functionality available in `verl/experimental/agent_loop/`

### Enhanced Type Checking
Added mypy configuration to `pyproject.toml` with selective type checking for critical modules

## Merge Code Rules

When merging code to the current branch, strictly遵守 the following rules:

1. Always run pre-commit checks before merging:
   ```bash
   pre-commit run --all-files
   ```

2. Ensure all tests pass:
   ```bash
   pytest tests/
   ```

3. Check for any new dependency conflicts:
   ```bash
   pip install -e .[test]
   ```

4. Verify documentation builds correctly:
   ```bash
   cd docs && make html
   ```

5. Follow the existing code style and patterns in the codebase

6. Ensure all new code has appropriate type hints

7. Update CLAUDE.md if new major features are added

8. When encountering code conflicts, carefully analyze the conflicting code functionality:
   - Situation 1: If the conflicting code implements the same functionality, update the current code to the incoming code and describe the functionality in the merge commit message
   - Situation 2: If the conflicting code implements different functionalities, retain the correctness and completeness of both implementations and add appropriate comments at the relevant code locations
   - Situation 3: When unable to determine how to handle conflicting code, be sure to retain the complete conflicting code and add descriptions of the code source and functionality at appropriate locations