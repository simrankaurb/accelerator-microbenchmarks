import argparse
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import json
import logging
import concurrent.futures
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting ICI Link Diagnostics")
    
    devices = jax.devices()
    num_devices = len(devices)
    logger.info(f"Detected {num_devices} devices")

    # 1. Build adjacency list based on coords (converting list to tuple for hashing)
    coords_to_dev = {tuple(dev.coords): dev for dev in devices}
    dev_to_index = {dev: i for i, dev in enumerate(devices)}
    
    max_coords = [0] * len(devices[0].coords)
    for dev in devices:
        for i, v in enumerate(dev.coords):
            max_coords[i] = max(max_coords[i], v)
            
    mesh_shape = tuple(v + 1 for v in max_coords)
    
    adjacent_pairs = []
    for dev in devices:
        c = tuple(dev.coords)
        for dim in range(len(c)):
            for delta in [-1, 1]:
                nc = list(c)
                nc[dim] = (nc[dim] + delta) % mesh_shape[dim]
                nc_tuple = tuple(nc)
                if nc_tuple in coords_to_dev:
                    neighbor = coords_to_dev[nc_tuple]
                    if dev.id != neighbor.id:
                        src_idx = dev_to_index[dev]
                        dst_idx = dev_to_index[neighbor]
                        adjacent_pairs.append((src_idx, dst_idx, tuple(dev.coords), tuple(neighbor.coords)))

    adjacent_pairs = list(set(adjacent_pairs))
    logger.info(f"Generated {len(adjacent_pairs)} directional links to test.")

    # 2. Prepare JAX Mesh and Data
    mesh = Mesh(devices, ('dev',))
    sharding = NamedSharding(mesh, P('dev'))

    # 10 MB payload
    payload_size = 10 * 1024 * 1024 // 4 
    x = jnp.ones((num_devices, payload_size), dtype=jnp.float32)
    x = jax.device_put(x, sharding)

    try:
        from jax.shard_map import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map

    import os
    simulate_src = os.environ.get('SIMULATE_HUNG_LINK_SRC')

    def test_link(src_idx, dst_idx):
        def _test_fn(data):
            if simulate_src is not None:
                # Simulate a physical hardware hang by sleeping longer than the ThreadPool timeout
                # We trigger on ANY tested link to guarantee a simulated failure
                import time
                time.sleep(15)
            return jax.lax.ppermute(data, axis_name='dev', perm=[(src_idx, dst_idx)])
            
        mapped_fn = shard_map(_test_fn, mesh=mesh, in_specs=P('dev'), out_specs=P('dev'))
        return jax.jit(mapped_fn)

    broken_links = []
    tested = 0

    # 3. Test links sequentially to isolate the exact failure
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for src_idx, dst_idx, src_c, dst_c in adjacent_pairs:
            fn = test_link(src_idx, dst_idx)
            
            def run_fn():
                res = fn(x)
                res.block_until_ready()
            
            future = executor.submit(run_fn)
            try:
                # Generous timeout for a 10MB P2P transfer
                future.result(timeout=10)
                tested += 1
            except concurrent.futures.TimeoutError:
                logger.error(f"TIMEOUT on link {src_c} -> {dst_c}")
                broken_links.append({
                    "src_chip": str(src_c),
                    "dst_chip": str(dst_c),
                    "bandwidth_gbps": 0.0,
                    "reason": "TIMEOUT"
                })
                # TPU is now likely locked up by the hung collective. Break early.
                break

    logger.info("ICI diagnostics complete.")
    
    status = "OK" if len(broken_links) == 0 else "FAILED"
    result = {
        "status": status,
        "tested_links": tested,
        "broken_links": len(broken_links),
        "details": broken_links
    }

    # CHS Ramble application parses this exact prefix
    print(f"ICI_DIAGNOSTICS_RESULT: {json.dumps(result)}")

if __name__ == "__main__":
    main()
