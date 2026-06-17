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

def worker_main(skip_links):
    logger.info("Starting ICI Link Diagnostics Worker")
    
    devices = jax.devices()
    num_devices = len(devices)
    logger.info(f"Detected {num_devices} devices")

    # 1. Build adjacency list based on coords
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

    # Filter out known broken links
    filtered_pairs = []
    for pair in adjacent_pairs:
        link_str = f"{pair[2]}->{pair[3]}"
        if link_str not in skip_links:
            filtered_pairs.append(pair)
            
    logger.info(f"Testing {len(filtered_pairs)} links after skipping {len(skip_links)} known broken links.")

    if not filtered_pairs:
        print("WORKER_SUCCESS")
        import os
        os._exit(0)

    # 2. Prepare JAX Mesh and Data
    mesh = Mesh(devices, ('dev',))
    sharding = NamedSharding(mesh, P('dev'))

    payload_size = 10 * 1024 * 1024 // 4 
    x = jnp.ones((num_devices, payload_size), dtype=jnp.float32)
    x = jax.device_put(x, sharding)

    try:
        from jax.shard_map import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map

    import os
    simulate_src = os.environ.get('SIMULATE_HUNG_LINK_SRC')
    simulate_dst = os.environ.get('SIMULATE_HUNG_LINK_DST')

    def test_link(src_idx, dst_idx, src_c, dst_c):
        def _test_fn(data):
            if simulate_src is not None and str(src_c) == simulate_src and str(dst_c) == simulate_dst:
                import time
                time.sleep(15)
            return jax.lax.ppermute(data, axis_name='dev', perm=[(src_idx, dst_idx)])
            
        mapped_fn = shard_map(_test_fn, mesh=mesh, in_specs=P('dev'), out_specs=P('dev'))
        return jax.jit(mapped_fn)

    tested = 0

    # 3. Warmup XLA runtime
    logger.info("Warming up XLA runtime...")
    jax.device_put(jnp.ones(1), jax.devices()[0]).block_until_ready()
    test_link(0, 0, (0,), (0,))(x).block_until_ready()
    logger.info("Warmup complete.")

    # 4. Test links sequentially
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for src_idx, dst_idx, src_c, dst_c in filtered_pairs:
            fn = test_link(src_idx, dst_idx, src_c, dst_c)
            
            def run_fn():
                res = fn(x)
                res.block_until_ready()
            
            future = executor.submit(run_fn)
            try:
                future.result(timeout=10)
                tested += 1
            except concurrent.futures.TimeoutError:
                logger.error(f"TIMEOUT on link {src_c} -> {dst_c}")
                print(f"WORKER_TESTED: {tested}")
                broken_json = json.dumps({
                    "src_chip": str(src_c),
                    "dst_chip": str(dst_c),
                    "bandwidth_gbps": 0.0,
                    "reason": "TIMEOUT"
                })
                print(f"WORKER_TIMEOUT_JSON: {broken_json}")
                print(f"WORKER_TIMEOUT_LINK: {src_c}->{dst_c}")
                # Exit immediately to avoid XLA thread hang
                import os
                os._exit(1)

    print(f"WORKER_TESTED: {tested}")
    print("WORKER_SUCCESS")
    import os
    os._exit(0)


def master_main():
    import sys
    import subprocess
    import os
    
    logger.info("Starting ICI Diagnostics Master")
    skip_links = []
    broken_links = []
    total_tested = 0
    
    while True:
        logger.info(f"Spawning worker (skipping {len(skip_links)} links)")
        cmd = [sys.executable, sys.argv[0], "--worker", "--skip-links", json.dumps(skip_links)]
        
        process = subprocess.Popen(cmd, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        worker_success = False
        worker_timeout_link = None
        worker_broken_json = None
        
        for line in process.stdout:
            print(f"[Worker] {line}", end='')
            if line.startswith("WORKER_TESTED:"):
                tested_this_run = int(line.split("WORKER_TESTED:")[1].strip())
                total_tested += tested_this_run
            elif line.startswith("WORKER_TIMEOUT_LINK:"):
                worker_timeout_link = line.split("WORKER_TIMEOUT_LINK:")[1].strip()
            elif line.startswith("WORKER_TIMEOUT_JSON:"):
                worker_broken_json = json.loads(line.split("WORKER_TIMEOUT_JSON:")[1].strip())
            elif line.startswith("WORKER_SUCCESS"):
                worker_success = True
                
        process.wait()
        
        if worker_timeout_link and worker_broken_json:
            logger.info(f"Master recorded broken link: {worker_timeout_link}")
            skip_links.append(worker_timeout_link)
            broken_links.append(worker_broken_json)
        elif worker_success:
            logger.info("Worker completed all remaining links successfully.")
            break
        else:
            logger.error("Worker died unexpectedly without reporting a timeout.")
            break

    status = "OK" if len(broken_links) == 0 else "FAILED"
    result = {
        "status": status,
        "tested_links": total_tested,
        "broken_links": len(broken_links),
        "details": broken_links
    }

    print(f"ICI_DIAGNOSTICS_RESULT: {json.dumps(result)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--skip-links", type=str, default="[]")
    args = parser.parse_args()
    
    if args.worker:
        worker_main(json.loads(args.skip_links))
    else:
        master_main()
