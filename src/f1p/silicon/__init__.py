"""M6 silicon runtime (plan §11a): device discovery, QoS lanes, the unified-memory budget,
accelerated kernels and the benchmark suite.

Everything here degrades gracefully: on Linux or Intel the same calls run on the CPU, so
the rest of the code never branches on the platform.
"""
