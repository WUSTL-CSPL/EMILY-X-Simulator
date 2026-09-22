# Packaging validation — September 22, 2026

The original workspace was retained. The unified generator, compatibility entry
point, splitter and demonstration wrapper were copied without changing their
simulation logic. The packaged dependency is the same upstream commit.

Checks performed:

- Downloaded the upstream submodule from its public HTTPS URL and checked out
  `a6124824f653467173c845c4ff609d4d35d6902f`.
- Checked the simulator, loaded IQ and reference terrain hashes.
- Generated all 4,000 scenes from this new directory, then created the
  1,000/200/200 geographic split. All **22 reference files matched byte-for-byte**.
- Ran 24 tests: **23 passed** and the historical v1-versus-v2 comparison was
  skipped because its archived input datasets are not part of this repository.
- Exported the staged Git files into an empty temporary directory, initialized
  the public submodule, and ran the tests available without IQ/terrain/datasets.
  Data-dependent tests were skipped as expected.
- Checked shell syntax, document links and Git exclusions. Local IQ/terrain
  links, datasets, logs and virtual environments are absent from the staged files.

The full pool's simulation/export phase took 146.7 seconds with eight workers;
that counter excludes startup and template construction. The earlier 155-second
measurement in the README covers the full generation command.

Validation reused the original workspace's Python 3.12 environment and linked
its IQ/terrain inputs. A new pip installation was not performed during packaging.
`setup_simulator.sh` provides that step for another checkout. No model fitting or
test-set scoring was performed, and nothing was pushed to GitHub.

Local generated reports are in `runs/reference_check.json`,
`runs/package_tests.log` and `runs/clean_checkout_check.json`; these logs are ignored
by Git. Repeat the commands in the README to create your own reports.
