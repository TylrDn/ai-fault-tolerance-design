# ai-fault-tolerance-design

Design document and simulator for fault tolerance in distributed training systems.

## Quick links

| Resource | Description |
|----------|-------------|
| [docs/design.md](docs/design.md) | Architecture design doc (failures, redundancy, recovery, chaos tests) |
| [src/simulator.py](src/simulator.py) | Python fault-injection simulator |
| [notebooks/01_fault_demo.ipynb](notebooks/01_fault_demo.ipynb) | Interactive demo notebook |

## Getting started

```bash
pip install -r requirements.txt

# Run the simulator (8 nodes, 5 % per-step failure rate, 200 steps)
python src/simulator.py --nodes 8 --fail-rate 0.05 --steps 200

# Run the test suite
pytest src/tests/ -v
```

## CI

GitHub Actions runs the test suite on every push – see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
