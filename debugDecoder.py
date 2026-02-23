"""
Debug: Check what data is available in SCL decoder after decoding.
"""
import numpy as np
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message

codec = SCListPolarCodec(N=128, K=64, L=4)
bpsk = SimpleBPSKModulationAWGN(fec_rate=64/128)

msg = generate_binary_message(size=64)
encoded = codec.encode(msg)
received = bpsk.transmit(message=encoded, snr_db=2.0)
decoded = codec.decode(received)

decoder = codec.decoder
print(f"Decoder type: {type(decoder)}")
print(f"Decoder attributes: {[x for x in dir(decoder) if not x.startswith('_')]}")
print(f"Has 'paths': {hasattr(decoder, 'paths')}")
if hasattr(decoder, 'paths'):
    print(f"Number of paths: {len(decoder.paths)}")
    if len(decoder.paths) > 0:
        path = decoder.paths[0]
        print(f"\nPath type: {type(path)}")
        print(f"Path attributes: {[x for x in dir(path) if not x.startswith('_') and not callable(getattr(path, x))]}")
        
        # Check some specific attributes
        for attr in ['intermediate_llr', 'current_llr', 'intermediate_bits', 'mask', 'N']:
            if hasattr(path, attr):
                val = getattr(path, attr)
                print(f"Path.{attr}: {type(val).__name__} (len={len(val) if isinstance(val, (list, np.ndarray)) else 'N/A'})")
