You are a planning/research agent. Create a precise, implementation-ready plan for roadmap step 14 in the `msbg-rs` repository.

First, inspect the repository—especially `docs/`—to locate the roadmap and identify the exact scope, acceptance criteria, dependencies, and wording for step 14. Do not infer step 14 from this prompt; treat the repository documentation as authoritative.

Project context:
- `msbg-rs` is a Rust reimplementation of the MSBG C++ library.
- The goal is cross-compatibility with the original where applicable, while removing dead code, improving code quality with idiomatic Rust, reducing unnecessary code, improving the public API, and meeting or exceeding original performance.
- The C++ source contains dead/disabled material (including `#if 0`-style content); distinguish executable/relevant behavior from inactive code.
- Solver details may be absent from the C++ source. The paper PDF within the MSBG source repository is authoritative for solver behavior when source code is incomplete.
- Do not propose mocked substitutes for original MSBG functionality when evaluating compatibility or performance.

Useful equations extracted from the paper:
% ==========================================
% 1. Governing Equations[cite: 1]
% ==========================================

% Momentum Equation (Eq. 1)[cite: 1]
\frac{\partial u}{\partial t}=-u\cdot\nabla u-\frac{\nabla p}{\rho}+v\nabla^{2}u+\frac{f}{\rho}

% Incompressibility Condition (Eq. 2)[cite: 1]
\nabla\cdot u=0

% Density Interpolation[cite: 1]
\rho(x)=\text{lerp}(\rho_{l},\rho_{g},\phi(x))

% ==========================================
% 2. Particle-to-Grid Transfer (P2G)[cite: 1]
% ==========================================

% Particle Mass (Eq. 5)[cite: 1]
m_{p} = \begin{cases} 
\rho_{l}V_{p} & \text{if } p \text{ is a liquid particle} \\ 
\rho_{g}V_{p} & \text{if } p \text{ is an air particle} 
\end{cases}

% Splatting Kernel Weight (Eq. 6)[cite: 1]
w_{ai,p}=(\max(1-(||x_{i}+e_{a}/2-x_{p}||/r_{p})^{2},0))^{3}

% Face Mass (Eq. 3)[cite: 1]
m_{ai}=\sum_{p}w_{ai,p}m_{p}

% Face Momentum (Eq. 4)[cite: 1]
P_{ai}=\sum_{p}w_{ai,p}m_{p}u_{p}

% Preliminary Cell Face Velocities[cite: 1]
\tilde{u}_{ai}^{*}=P_{ai}/m_{ai}

% ==========================================
% 3. Phase Field Calculation[cite: 1]
% ==========================================

% Raw Mass Densities[cite: 1]
\tilde{\rho}_{ai}=m_{ai}

% Phase Field Normalization and Clamping (Eq. 7)[cite: 1]
\phi(\tilde{\rho}) = \begin{cases} 
0 & \text{if } \tilde{\rho} < \tilde{\rho}_{min} \\ 
\min\left(\max\left(\frac{\tilde{\rho}-\tilde{\rho}_{min}}{\alpha_{\phi}\rho_{l}\tilde{\rho}_{0}}, 0\right), 1\right) & \text{else} 
\end{cases}

% ==========================================
% 4. Pressure Projection[cite: 1]
% ==========================================

% Divergence-Free Velocity Update[cite: 1]
u=u^{*}-\Delta t\frac{1}{\rho}\nabla p

% Pressure Poisson Equation (Eq. 8)[cite: 1]
\Delta t\nabla\cdot(\beta(x)\nabla p(x)) = \nabla\cdot u(x) + c_{div}(x)

% Face Coefficients (Eq. 9)[cite: 1]
\beta(x)=1/\rho(x)

% 1D Discretization on a MAC Grid (Eq. 10)[cite: 1]
\beta_{i+\frac{1}{2}}\left(\frac{p_{i+1}-p_{i}}{\Delta x}\right)-\beta_{i-\frac{1}{2}}\left(\frac{p_{i}-p_{i-1}}{\Delta x}\right)=\frac{1}{\Delta t}(u_{i+\frac{1}{2}}^{*}-u_{i-\frac{1}{2}}^{*})

% Linear System Representation[cite: 1]
\mathcal{L}p=\frac{1}{\Delta t}D(u^{*})

% ==========================================
% 5. Grid-to-Particle Transfer (G2P) & Viscosity[cite: 1]
% ==========================================

% Overall Velocity Change (Eq. 11)[cite: 1]
\Delta u=u-\tilde{u}^{*}

% FLIP Velocity Update (Eq. 12)[cite: 1]
u_{p}^{new}=\alpha_{FLIP}(u_{p}^{old}+I(\Delta u,x_{p}))+(1-\alpha_{FLIP})I(u,x_{p})

% Artificial / Numerical Viscosity (Eq. 13)[cite: 1]
v = (1-\alpha_{FLIP})\frac{\Delta x^{2}}{6\Delta t}

% ==========================================
% 6. Spray Particles[cite: 1]
% ==========================================

% Physics-Based Evaporation ODE[cite: 1]
\frac{dr_{p}}{dt} \sim -1/r

Your output must be a plan only: do not implement code, modify files, or claim unverified behavior.

Research requirements:
1. Map step 14 to the existing Rust architecture, current roadmap state, adjacent completed/pending steps, relevant source files, tests, fixtures, build scripts, and benchmarks.
2. Locate and inspect the original C++ implementation and paper PDF available in the workspace/repository. Identify the executable/reference behavior relevant to step 14, separating it from disabled/dead code.
3. If the step concerns solver behavior, derive implementation requirements from both the active C++ behavior and the paper, explicitly identifying any ambiguity or missing source behavior.
4. Identify public API implications, compatibility expectations, numerical invariants, error handling, ownership/layout concerns, and likely Rust-idiomatic simplifications.

Testing requirements:
- Before proposing individual tests, provide a test matrix.
- The matrix must cover dimensions such as:
  - normal cases (only one or two happy paths total);
  - boundary/grid-edge conditions;
  - empty, singleton, degenerate, and minimum-size inputs;
  - awkward calling conventions and API misuse/error paths;
  - numerical edge cases (zero/near-zero values, thresholds, clamping, density contrasts, timestep limits, non-finite inputs if the API permits them);
  - conservation/invariant checks where relevant;
  - C++-vs-Rust differential or golden comparisons using real MSBG behavior;
  - regression cases for known or discovered quirks.
- For each matrix row, state the purpose, setup, oracle, and whether it is unit, integration, differential, property, or regression coverage.
- Only after the matrix, identify the concrete tests to write. Avoid shallow happy-path-only tests.

Performance-completion gate:
- The plan must define realistic, reproducible benchmarks that compare Rust directly against the original C++ MSBG implementation using equivalent real inputs, configurations, solver primitives, and execution paths.
- Do not use mocked functions, stand-ins, synthetic replacement algorithms, or benchmark-only simplified implementations.
- Make comparisons apples-to-apples: same domain/grid resolution, particle distributions, phase ratios, timesteps, solver parameters/tolerances, warm-up policy, repetitions, compiler optimization modes, and measurement methodology.
- Include both runtime/latency and throughput metrics where meaningful, plus memory/allocation behavior if measurable without distorting results.
- Specify correctness validation before timing so a faster incorrect implementation cannot pass.
- Define explicit pass/fail acceptance criteria for “Rust is superior,” including how to treat noise and regressions.
- Explain benchmark artifact locations, commands, result formats, baseline capture, and CI versus dedicated-performance-runner responsibilities.

Produce the final plan with:
1. “Step 14 interpretation” — quoted/paraphrased roadmap requirement with source paths.
2. “Current-state findings” — concise evidence-backed inventory of relevant code and reference material.
3. “Design decisions and compatibility contract.”
4. “Test matrix” — required before enumerating tests.
5. “Implementation plan” — ordered, file-level steps; each includes target files, intended changes, rationale, dependencies, and validation.
6. “Benchmark and performance gate.”
7. “Risks, ambiguities, and decisions requiring confirmation.”

Use exact repository paths and symbols discovered during inspection. Clearly label facts verified from the repository versus inferences from the paper/equations. Do not fabricate paths, APIs, benchmark results, or C++ behavior. This is not a 1:1 port internally, we want to be BETTER. If you find a regression, you must demonstrate WHY using the tools available in the nix shell files, and then fix those regressions. Only parity or surpassing is acceptable.
