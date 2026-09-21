# Contributing to hIPPyMFEM

Contributions are welcome at every level: bug reports, fixes, new capabilities, better
documentation, and new tutorials or applications. For a large change, please open an
issue first, so that the design can be discussed before the work is done.

These guidelines are adapted from those of hIPPYlib and MFEM.

## Workflow

1. Fork the repository and create a branch off `main`, named for what it does
   (`anisotropic-robin-dev`, `ghost-dof-fix`).
2. Make the change together with its tests and documentation (see the checklist below).
3. Run the test suites on more than one rank.
4. Open a pull request against `main`: say what changed and why, and list anything still
   to do. Please allow maintainers to push to your branch.

Continuous integration must pass before a pull request is merged.

## Developer guidelines

**Keep the code lean.** New features go in when they are general or clearly useful;
problem-specific code belongs in `applications/`. The inverse-problem classes keep
hIPPYlib's names and signatures where there is a counterpart
(`docs/source/correspondence.rst`); keep that when extending them.

**Test against something external.** Every suite in `hippymfem/test` holds the library
against an independent reference: MFEM's own integrators, finite differences, dense linear
algebra, exact Gaussian posteriors, or hIPPYlibx on a shared discrete problem. A test that
checks the code only against itself misses the errors that matter. The suites are MPI
programs, so a check must reach the same verdict on every rank.

**Parallel correctness.** Most bugs that pass on one rank are one of these:

- a decision that can differ between ranks (a loop count, an early return, a branch on a
  local size) taken before a collective call, which deadlocks the job;
- ghost degrees of freedom treated as owned ones, or a local size used where the global
  size is meant;
- a scalar summed on each rank and never reduced.

Run new code on 1, 2 and 4 ranks, with `./run_tests.sh 1 2 4` or
`HIPPYMFEM_TEST_RANKS=1,2,4 python -m pytest`. `tools/check_collectives.py` scans for the
most common of these patterns.

**Object lifetime.** PyMFEM hands raw pointers to MFEM, so a Python object that is garbage
collected too early is a crash, not an exception. Keep alive whatever you pass to MFEM
(`KeepAlive`, `hippymfem.fem.assemble.own`), and use `hm.to_numpy` rather than the views
`GetDataArray()` returns.

**Documentation.** Public classes and functions have NumPy-style docstrings, from which the
API reference is generated. A new module gets an entry under `docs/source/api/`.

**Style.** Follow PEP 8 and the conventions of the surrounding code. Comments explain why,
not what.

## Pull request checklist

- [ ] The test suites pass on 1, 2 and 4 ranks.
- [ ] New functionality is tested against an external reference.
- [ ] Changes to the public API are documented: docstrings, `docs/source/api/`, and the
      user guide where relevant.
- [ ] The tutorials and applications still run (`.github/workflows/run_notebooks.sh` and
      `.github/workflows/run_applications.sh`).
- [ ] `CHANGELOG.md` has an entry for anything a user would notice.

## Automated testing

GitHub Actions runs on every push to `main` and on every pull request:

- **pymfem** builds PyMFEM with MPI and caches the wheel, so later runs skip the build;
- **tests** runs every suite on 1 and 2 ranks through pytest;
- **notebooks** executes every tutorial;
- **applications** runs each application on small inputs, on 1 and 2 ranks;
- **docs** builds the documentation without PyMFEM, as Read the Docs does.

The GPU suites (`test_gpu`, `test_device`) need hardware that CI does not have. Run them
with `run_tests.sh` on a machine with a GPU (see `INSTALL.md`).

## Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit
it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is
covered under an appropriate open source license and I have the right under that license to
submit that work with modifications, whether created in whole or in part by me, under the
same open source license (unless I am permitted to submit under a different license), as
indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a),
(b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a
record of the contribution (including all personal information I submit with it, including
my sign-off) is maintained indefinitely and may be redistributed consistent with this
project or the open source license(s) involved.
