.PHONY: ci-all ci-test

ci-all:
	@echo "Connectomics CI pipeline (U1→U10)"
	@echo "-----------------------------------"
	@echo "U1  Configuration bootstrap"
	@echo "U2  Dependency environment"
	@echo "U3  Static typing (stubs in src/cx_connectome/ci)"
	@echo "U4  Config defaults (configs/ci.yaml)"
	@echo "U5  Simulation scaffolding"
	@echo "U6  Integration plumbing"
	@echo "U7  Metric hooks"
	@echo "U8  Optimization interface"
	@echo "U9  Dynamics + RFC alignment"
	@echo "U10 Test orchestration"
	@echo "Use 'make ci-test' to run the dedicated CI test suite."

ci-test:
	@python -m pytest tests/ci
