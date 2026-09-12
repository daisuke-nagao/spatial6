# ABA comparison measurement report

This report covers one full run of:

```text
cargo bench --bench vs_featherstone -- aba_vs_featherstone_f32
```

The measured source commit was `d8af1256f1b14c247968c4acb84d67825c8ad85c`.
All 20 cases completed after the untimed correctness preflight. Values below
are Criterion slope point estimates with 95% confidence intervals. Times are
per complete solve in microseconds. The ratio is explicitly **Featherstone /
spatial6**; a ratio above 1 means the spatial6 estimate is lower for that case.

| Family | Links | spatial6 Builtin scalar alloc | Featherstone 0.1.0 alloc | Featherstone / spatial6 |
| --- | ---: | ---: | ---: | ---: |
| `planar_z` | 2 | 0.97779 [0.96893, 0.98719] | 1.60161 [1.53054, 1.67903] | 1.64 |
| `planar_z` | 4 | 2.33699 [2.30156, 2.37502] | 3.03391 [2.96998, 3.10114] | 1.30 |
| `planar_z` | 8 | 4.17403 [4.08775, 4.27930] | 6.23699 [6.16287, 6.31841] | 1.49 |
| `planar_z` | 16 | 8.66783 [8.45921, 8.86865] | 12.48076 [12.33553, 12.64152] | 1.44 |
| `planar_z` | 32 | 15.93494 [15.77857, 16.09301] | 27.20638 [26.92778, 27.47821] | 1.71 |
| `spatial_xyz` | 2 | 1.04789 [1.03091, 1.06554] | 1.57220 [1.53679, 1.60450] | 1.50 |
| `spatial_xyz` | 4 | 1.96706 [1.93427, 1.99864] | 3.12649 [3.04760, 3.21455] | 1.59 |
| `spatial_xyz` | 8 | 4.05331 [3.99478, 4.11824] | 6.27285 [6.16732, 6.39381] | 1.55 |
| `spatial_xyz` | 16 | 8.23020 [8.12116, 8.35677] | 12.91106 [12.61445, 13.22301] | 1.57 |
| `spatial_xyz` | 32 | 16.08861 [15.95353, 16.24327] | 28.20901 [27.52561, 28.97192] | 1.75 |

## Workload and timing boundary

The comparison uses fixed-base serial chains with one scalar revolute DoF per
link, `f32` on both sides, Builtin spatial6 storage, and the `planar_z` and
`spatial_xyz` families at 2, 4, 8, 16, and 32 links. Each timed call includes
q-dependent transform construction, per-call temporary solver-workspace
allocation, and all three ABA passes. Both implementations allocate temporary
solver workspace per call. Model construction, input installation, correctness
preflight, diagnostics, and reference solves are outside the timed closure.

The spatial6 implementation is benchmark/test support only. It uses checked
public `ArticulatedBodyInertia` primitives and dynamic `Vec` storage, while
Featherstone uses general-joint dynamic storage. These specialization, storage,
checking, and allocation differences are part of the measured workload; the
result is not an isolated ABI-storage measurement.

The existing `rnea_vs_featherstone` group is excluded. It remains the legacy
spatial6 `f64`/Featherstone `f32` comparison with differing gravity settings.
Its timings must not be compared with this ABA group, and neither group supports
a general whole-crate speed claim.

## Correctness observations

- All 20 selected cases passed the untimed preflight.
- The maximum observed output error against the independent dense `f64` oracle
  was `5.5904747e-5` (Featherstone versus oracle, `planar_z`, 8 links).
- The spatial6 preflight checked every runtime pivot against `d_i >= 1e-4`.
  Independently, reverse elimination of the dense reference mass matrix gave
  at least `7.499999963e-3` for all `planar_z` lengths. For `spatial_xyz`, the minimum was
  `3.500000004e-3` at 4 and 16 links; the other lengths were
  `6.900000247e-3`. The overall minimum was therefore about 35 times the
  `1e-4` acceptance threshold.

## Reproduction metadata

- Cargo.lock SHA-256:
  `111D63E9AD40DE8665B13129085A722F64E71375FC486C8CA5467E36C0FB31C3`
- `featherstone` 0.1.0 crates.io checksum:
  `622260d3641d32d8f5e4d402a512fc0ef3e69dcea5fbd9701bdade8841ef2850`
- Toolchain: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, `cargo 1.97.1`.
- Target: `x86_64-pc-windows-msvc`; optimized benchmark profile;
  `RUSTFLAGS` unset.
- Features: spatial6 default Builtin `f32`; Featherstone `serde,tracing`.
- Criterion: 0.8.2, 3-second warm-up, 100 samples with approximately
  5-second collection; Plotters fallback because gnuplot was unavailable.
- Machine: AMD Ryzen AI 9 HX 370 with Radeon 890M, 24 logical processors;
  Windows `10.0.26200 X64`.
- Conditions: Balanced power plan; no concurrent cargo/rustc work, ordinary
  Codex/Windows background processes, and no CPU pinning or isolation.

The sampled points are near-linear over these chain lengths, with ratios from
1.30 to 1.75. This is a single-machine, single-run observation, not an
asymptotic result or a general performance claim about either crate.
