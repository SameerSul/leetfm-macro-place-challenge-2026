# VivaPlace Research and External References

This document records the research papers, technical sources, benchmark sources,
and principal external software links that directly inform VivaPlace. It is an
attribution and provenance index, not a claim that every cited algorithm is
enabled in production.

Status terms used below:

- **Production**: the active hierarchy-only placement or evaluation path uses
  the cited implementation or technique.
- **Basis**: the work defines a benchmark, baseline, objective, or upstream
  algorithm on which the program depends.
- **Research-only**: code, data, or documentation remains in the repository but
  the technique is default-off or offline-only.
- **Historical**: the technique was evaluated and rejected or its code was
  removed; it remains cited because the experiment is recorded in
  `PROGRESS.md`.
- **Future reference**: the paper is named in an open issue but its method is
  not implemented.

## Production Placement Papers

1. **DREAMPlace — production / direct dependency.** Yibo Lin, Shounak Dhar,
   Wuxi Li, Haoxing Ren, Brucek Khailany, and David Z. Pan, “DREAMPlace: Deep
   Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement,” DAC
   2019. [DOI](https://doi.org/10.1145/3316781.3317803),
   [author preprint](https://yibolin.com/publications/papers/PLACE_DAC2019_Lin.pdf),
   [extended TCAD paper](https://doi.org/10.1109/TCAD.2020.3003843).
   VivaPlace uses the pinned DREAMPlace global placer through
   `src/dreamplace_bridge/`; DREAMPlace legalization and detailed placement are
   disabled.

2. **DREAMPlace 4.1 / second-order backbone — production / direct
   dependency.** Yifan Chen, Zaiwen Wen, Yun Liang, and Yibo Lin, “Stronger
   Mixed-Size Placement Backbone Considering Second-Order Information,” ICCAD
   2023. [DOI](https://doi.org/10.1109/ICCAD57390.2023.10323700),
   [author preprint](https://yibolin.com/publications/papers/PLACE_ICCAD2023_Chen.pdf).
   The grouped seed stage sets `macro_place_flag=1` and `use_bb=1`, selecting
   DREAMPlace 4.1's short Barzilai-Borwein-scaled Nesterov update.

3. **ePlace — basis through DREAMPlace.** Jingwei Lu, Pengwen Chen, Chin-Chih
   Chang, Lu Sha, Dennis Jen-Hsin Huang, Chin-Chi Teng, and Chung-Kuan Cheng,
   “ePlace: Electrostatics-Based Placement Using Fast Fourier Transform and
   Nesterov's Method,” ACM TODAES 20(2), 2015.
   [DOI](https://doi.org/10.1145/2699873),
   [author PDF](https://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf).
   This is the electrostatic analytical-placement foundation inherited through
   DREAMPlace; VivaPlace does not separately reimplement ePlace.

4. **Nesterov acceleration — production through DREAMPlace.** Yurii E.
   Nesterov, “A Method of Solving a Convex Programming Problem with Convergence
   Rate O(1/k^2),” *Doklady Akademii Nauk SSSR* 269(3), 1983, pp. 543–547.
   [MathNet record](https://www.mathnet.ru/eng/dan46009).

5. **Barzilai-Borwein step size — production through DREAMPlace 4.1.**
   Jonathan Barzilai and Jonathan M. Borwein, “Two-Point Step Size Gradient
   Methods,” *IMA Journal of Numerical Analysis* 8(1), 1988, pp. 141–148.
   [DOI](https://doi.org/10.1093/imanum/8.1.141).

6. **IncreMacro — production technique reference.** Yuan Pu, Tinghuan Chen,
   Zhuolun He, Chen Bai, Haisheng Zheng, Yibo Lin, and Bei Yu, “IncreMacro:
   Incremental Macro Placement Refinement,” ISPD 2024, pp. 169–176.
   [DOI](https://doi.org/10.1145/3626184.3633321),
   [author PDF](https://www.cse.cuhk.edu.hk/~byu/papers/C205-ISPD2024-IncreMacro.pdf).
   VivaPlace's `src/placer/legalize/constraint_graph.py` is an independent,
   bounded H/V separation-DAG implementation in the same constraint-graph
   legalization family; it does not reproduce IncreMacro's LP, diagnosis,
   macro-shifting, or cell-migration flow.

7. **PeF — supporting constraint-graph reference.** Ximeng Li, Keyu Peng,
   Fuxing Huang, and Wenxing Zhu, “PeF: Poisson's Equation Based Large-Scale
   Fixed-Outline Floorplanning,” arXiv:2210.03293, 2022.
   [arXiv](https://arxiv.org/abs/2210.03293). PeF is a second literature example
   of horizontal/vertical constraint-graph overlap legalization; its Poisson
   floorplanner is not implemented here.

8. **Circuit Training objective lineage — basis.** Azalia Mirhoseini, Anna
   Goldie, Mustafa Yazgan, Joe Jiang, Ebrahim Songhori, Shen Wang, Young-Joon
   Lee, Eric Johnson, Omkar Pathak, Azade Nazi, Jiwoo Pak, Andy Tong, Kavya
   Srinivasa, William Hang, Emre Tuncer, Quoc V. Le, James Laudon, Richard Ho,
   Roger Carpenter, and Jeff Dean, “A Graph Placement Methodology for Fast Chip
   Design,” *Nature* 594, 2021, pp. 207–212.
   [DOI](https://doi.org/10.1038/s41586-021-03544-w),
   [Circuit Training code](https://github.com/google-research/circuit_training).
   The TILOS evaluator used by this challenge reproduces this work's normalized
   wirelength, density, and congestion proxy; VivaPlace does not run the RL
   policy.

9. **TILOS MacroPlacement assessment — production evaluator / basis.**
   Chung-Kuan Cheng, Andrew B. Kahng, Sayak Kundu, Yucheng Wang, and Zhiang
   Wang, “Assessment of Reinforcement Learning for Macro Placement,” ISPD
   2023, pp. 158–166. [DOI](https://doi.org/10.1145/3569052.3578926),
   [arXiv](https://arxiv.org/abs/2302.11014),
   [author PDF](https://vlsicad.ucsd.edu/Publications/Conferences/396/c396.pdf).
   This work and its repository supply the exact `PlacementCost` evaluator and
   benchmark infrastructure used by the program.

10. **Updated TILOS assessment — current challenge basis.** Chung-Kuan Cheng,
    Andrew B. Kahng, Sayak Kundu, Yucheng Wang, and Zhiang Wang, “An Updated
    Assessment of Reinforcement Learning for Macro Placement,” IEEE TCAD,
    2025 early access. [DOI](https://doi.org/10.1109/TCAD.2025.3644293),
    [author PDF](https://vlsicad.ucsd.edu/Publications/Journals/j148.pdf).
    The Partcl/HRT challenge uses the baselines and modern-design evaluation
    context from this update.

11. **RePlAce — benchmark baseline / DREAMPlace comparison basis.**
    Chung-Kuan Cheng, Andrew B. Kahng, Ilgweon Kang, and Lutong Wang,
    “RePlAce: Advancing Solution Quality and Routability Validation in Global
    Placement,” IEEE TCAD 38(9), 2019, pp. 1717–1730.
    [DOI](https://doi.org/10.1109/TCAD.2018.2859220),
    [author PDF](https://vlsicad.ucsd.edu/Publications/Journals/j126.pdf),
    [code](https://github.com/The-OpenROAD-Project/RePlAce). RePlAce is a score
    baseline; its placer is not called by VivaPlace.

12. **IBM fixed-outline benchmark lineage — basis.** Saurabh N. Adya and Igor
    L. Markov, “Fixed-Outline Floorplanning: Enabling Hierarchical Design,”
   IEEE TVLSI 11(6), 2003, pp. 1120–1135.
   [DOI](https://doi.org/10.1109/TVLSI.2003.817546).

13. **ICCAD04 mixed-size benchmark lineage — basis.** Saurabh N. Adya,
    S. Chaturvedi, J. Roy, D. A. Papa, and Igor L. Markov, “Unification of
    Partitioning, Placement and Floorplanning,” ICCAD 2004.
    [DOI](https://doi.org/10.1109/ICCAD.2004.1382639),
    [paper PDF](https://www.cs.york.ac.uk/rts/docs/SIGDA-Compendium-1994-2004/papers/2004/iccad04/pdffiles/07c_1.pdf).

14. **Rent-style synthetic locality — diagnostic benchmark basis.** Bernard S.
    Landman and Roy L. Russo, “On a Pin Versus Block Relationship for
    Partitions of Logic Graphs,” *IEEE Transactions on Computers* C-20(12),
    1971, pp. 1469–1479.
    [DOI](https://doi.org/10.1109/T-C.1971.223159). The synthetic suite uses
    “Rent-style” only as a qualitative locality pattern; it does not fit or
    enforce Rent parameters.

15. **BeyondPPA — research-only structural features.** Ishraq Tashdid,
    Valentina Terry, Jordan Merkel, Tasnuva Farheen, and Sazadur Rahman,
    “BeyondPPA: Human-Inspired Reinforcement Learning for Post-Route
    Reliability-Aware Macro Placement,” MLCAD 2025.
    [DOI](https://doi.org/10.1109/MLCAD65511.2025.11189164),
    [OpenReview PDF](https://openreview.net/pdf/7094d1eff97f5a5c69703ded8b9d79162c9c95ff.pdf).
    VivaPlace retains default-off deterministic relocation ordering features
    for I/O keepout, alignment, and notch avoidance. It does not run the
    paper's reinforcement-learning policy.

16. **MacroDiff+ — research-only data-schema inspiration.** Jongho Yoon,
    Jinsung Jeon, and Seokhyeong Kang, “Physics-Guided Geometric Diffusion for
    Macro Placement Generation,” arXiv:2605.16451, 2026.
    [arXiv](https://arxiv.org/abs/2605.16451). Only the heterogeneous macro-net
    graph view inspired the offline Stage-G4 dataset schema; no diffusion model
    or physics-guided sampler is used in placement. The repository URL stated
    by the paper returned `404` during this reference audit, so it is not
    presented as a working source link here.

17. **WireMask-BBO — historical / removed.** Yunqi Shi, Ke Xue, Lei Song, and
    Chao Qian, “Macro Placement by Wire-Mask-Guided Black-Box Optimization,”
    NeurIPS 2023. [arXiv](https://arxiv.org/abs/2306.16844),
    [code](https://github.com/lamda-bbo/WireMask-BBO). Constructive WireMask
    experiments regressed dense designs and were removed; only their results
    remain in `PROGRESS.md`.

18. **RUDY — historical / removed.** Peter Spindler and Frank M. Johannes,
    “Fast and Accurate Routing Demand Estimation for Efficient
    Routability-Driven Placement,” DATE 2007, pp. 1226–1231.
    [DOI](https://doi.org/10.1109/DATE.2007.364463).
    Deterministic RUDY-based area inflation was evaluated, rejected, and
    deleted; the exact TILOS congestion model remains active.

19. **Zhang-Hager non-monotone line search — historical / removed.** Hongchao
    Zhang and William W. Hager, “A Nonmonotone Line Search Technique and Its
    Application to Unconstrained Optimization,” *SIAM Journal on Optimization*
    14(4), 2004, pp. 1043–1056.
    [DOI](https://doi.org/10.1137/S1052623403428208). A paper-faithful bounded
    non-monotone Armijo trial was evaluated on ibm04 and ibm10, regressed
    DREAMPlace seed quality, and was removed; results remain in `PROGRESS.md`.

20. **ArchGen challenge write-up — historical technical source, not a paper.**
    ArchGen AI, “How We Ranked First in the HRT (Hudson River Trading) and
    Partcl Macro Placement Challenge,” June 20, 2026.
    [project and article page](https://www.archgen.tech/). Seed-portfolio,
    weighted-proposal, and buffered-telemetry experiments were described as
    ArchGen-inspired in the experiment ledger; this is not a peer-reviewed
    research citation or a code dependency.

## Project, Evaluator, Data, and Tool Links

| Resource | Link | Use in this repository |
| --- | --- | --- |
| Partcl/HRT Macro Placement Challenge | [challenge repository](https://github.com/partcleda/macro-place-challenge-2026) | Rules, harness, packaging, and leaderboard context |
| Partcl MacroPlacement fork | [submodule source](https://github.com/partcleda/MacroPlacement) | The checked-out `external/MacroPlacement` evaluator fork |
| TILOS MacroPlacement | [upstream repository](https://github.com/TILOS-AI-Institute/MacroPlacement) | Exact evaluator, testcases, enablements, and reproducibility material |
| TILOS proxy definition | [Proxy Cost documentation](https://tilos-ai-institute.github.io/MacroPlacement/Docs/ProxyCost/) | Wirelength, density, and congestion objective definition |
| TILOS `PlacementCost` implementation | [source file](https://github.com/TILOS-AI-Institute/MacroPlacement/blob/main/CodeElements/Plc_client/plc_client_os.py) | Scalar reference for scoring and verification |
| DREAMPlace | [repository](https://github.com/limbo018/DREAMPlace), [release 4.1.0](https://github.com/limbo018/DREAMPlace/releases/tag/4.1.0), [pinned upstream commit](https://github.com/limbo018/DREAMPlace/commit/37214b40fe3837cc7d392c7d6092ccd6ff04a02c) | Required global-placement engine and reproducible source pin |
| Circuit Training | [repository](https://github.com/google-research/circuit_training) | File-format and proxy-objective lineage; the RL placer is not run |
| OpenROAD | [repository](https://github.com/The-OpenROAD-Project/OpenROAD) | Open-source EDA-flow and NG45 interoperability context |
| FreePDK45 / NanGate45 | [NCSU FreePDK45 page](https://eda.ncsu.edu/freepdk/freepdk45/) | Open 45-nm enablement context for commercial-style tests |
| PyTorch CUDA 12.1 wheels | [package index](https://download.pytorch.org/whl/cu121) | Pinned DREAMPlace build environment |

## Direct Python and Build Dependencies

These are the principal direct packages and build tools declared by the root
project or the reproducible DREAMPlace bootstrap. Exact versions and artifact
hashes are authoritative in `pyproject.toml`, `uv.lock`,
`scripts/dreamplace/requirements.txt`, and
`scripts/dreamplace/environment.yml`.

| Category | Projects |
| --- | --- |
| Runtime | [Python](https://www.python.org/), [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/), [Numba](https://numba.pydata.org/), [Matplotlib](https://matplotlib.org/), [tqdm](https://tqdm.github.io/), [Abseil Python](https://github.com/abseil/abseil-py) |
| Optional ML and baselines | [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), [SciPy](https://scipy.org/) |
| Test and style | [pytest](https://pytest.org/), [pytest-cov](https://pytest-cov.readthedocs.io/), [Black](https://black.readthedocs.io/), [Flake8](https://flake8.pycqa.org/) |
| Environment and native build | [uv](https://docs.astral.sh/uv/), [micromamba](https://mamba.readthedocs.io/), [CMake](https://cmake.org/), [GCC](https://gcc.gnu.org/), [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) |

## Scope Boundary

This index intentionally does not copy every transitive citation or URL from
the vendored `external/MacroPlacement`, generated `dreamplace_src`,
`dreamplace_build`, or `uv.lock` trees. Those include hundreds of upstream
papers, package artifact URLs, badge links, and third-party implementation
references that VivaPlace does not directly select. For those complete
upstream inventories, see the [DREAMPlace publication
list](https://github.com/limbo018/DREAMPlace#publications), the
[TILOS MacroPlacement documentation](https://tilos-ai-institute.github.io/MacroPlacement/),
and the dependency lockfiles. Apache license URLs are licensing boilerplate,
not research references, and are likewise not duplicated here.
