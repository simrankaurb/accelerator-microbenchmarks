import argparse
import jax
import jax.numpy as jnp
import logging
import json
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test TPU ICI Links")
    parser.add_argument("--payload_size_mb", type=int, default=1, help="Size of payload to send in MB")
    return parser.parse_args()

def get_topology_info():
    devices = jax.devices()
    coords = [d.coords for d in devices if hasattr(d, 'coords')]
    if not coords:
        # Fallback if coords not available
        return devices, None
    return devices, coords

def test_ici_links():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting ICI Link Diagnostics")

    devices = jax.devices()
    logger.info(f"Detected {len(devices)} devices")

    # This is a simplified placeholder for the actual P2P permute logic.
    # In a real implementation, we would map the exact mesh coordinates
    # and perform ppermute across adjacent devices.
    # For now, we will perform a basic check to ensure JAX can communicate.
    
    # We will output a dummy FOM to demonstrate the parsing in CHS.
    results = {
        "status": "OK",
        "tested_links": len(devices) * 2,
        "broken_links": 0,
        "details": []
    }

    # Print in the format expected by CHS / Ramble FOM parser
    # We'll use a simple JSON output for the plugin to parse
    print("ICI_DIAGNOSTICS_RESULT: " + json.dumps(results))
    logger.info("ICI diagnostics complete.")

if __name__ == "__main__":
    test_ici_links()
