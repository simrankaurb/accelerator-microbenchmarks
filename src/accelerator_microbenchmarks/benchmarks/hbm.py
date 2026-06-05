"""HBM bandwidth microbenchmarks."""

from typing import Any
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp


@registry.benchmark_registry.register("hbm_bandwidth")
class HBMBandwidthBenchmark(base.BaseBenchmark):
  """HBM bandwidth microbenchmark using a simple memory-bound kernel (y = x + 0).

  This kernel reads from HBM and writes back to HBM, ensuring we capture
  the round-trip bandwidth.
  """

  def setup(self, **params):
    @jax.jit
    def identity_op(x):
      with jax.named_scope(constants.MARKER):
        # Simple element-wise op to ensure HBM traffic
        return x + 0

    self._jit_fn = identity_op

  def generate_inputs(self, **params) -> tuple[jnp.ndarray, ...]:
    # 'size' being the number of elements
    size = params.get("size", 1024 * 1024 * 128)  # Default ~256MB for float16
    dtype_str = params.get("dtype", "bfloat16")
    dtype = getattr(jnp, dtype_str) if hasattr(jnp, dtype_str) else jnp.bfloat16

    # Parallelize across the mesh to utilize all devices
    if not self.mesh or self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(self.mesh.axis_names[0])
    )

    # Use jit with out_shardings to generate on devices to avoid host OOM
    generate_data = jax.jit(
        lambda k: jax.random.normal(k, (size,), dtype=dtype),
        out_shardings=sharding,
    )
    key = jax.random.PRNGKey(0)
    x = generate_data(key)

    return (x,)

  def run_op(self, x) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x)

  def get_total_bytes(self, **params) -> float:
    size = params.get("size", 1024 * 1024 * 128)
    dtype_str = params.get("dtype", "bfloat16")
    dtype = getattr(jnp, dtype_str) if hasattr(jnp, dtype_str) else jnp.bfloat16
    itemsize = jnp.dtype(dtype).itemsize
    # 1 read (X) + 1 write (Y)
    return size * itemsize * 2

  def get_arithmetic_intensity(self, **params) -> float:
    # For HBM copy, intensity is almost 0.
    # But wait, 'x + 0' technically does 1 flop per element.
    # Arithmetic Intensity = Flops / Bytes
    size = params.get("size", 1024 * 1024 * 128)
    flops = size  # 1 add per element
    bytes_moved = self.get_total_bytes(**params)
    return flops / bytes_moved

  def calculate_metrics(
      self, times_ms: list[float], **params
  ) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    total_bytes = self.get_total_bytes(**params)

    avg_latency_s = metrics["avg_ms"] / 1000.0
    if avg_latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = (total_bytes / avg_latency_s) / 1e9

    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics["total_bytes_mb"] = total_bytes / 1e6
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics
