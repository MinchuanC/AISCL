from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)
import torch

N = 128
K = 64
design_snr = 0.0
messages = 1000
# SNR in [.0, .5, ..., 4.5, 5]
snr_range = [i / 2 for i in range(11)]

# Load trained AI model
ai_model = PathPruningNet(N)
ai_model.load_state_dict(torch.load('trained_model_N128_K64.pt'))
ai_model.eval()

# Use AI-based SCL codec with L=4
codec = AISCLPolarCodec(N=N, K=K, design_snr=design_snr, L=4, ai_model=ai_model)
bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

result_ber = dict()
result_fer = dict()

print('Python polar coding simulation')
print(f'Simulating ({codec.N}, {codec.K}) systematic polar code with Design SNR {codec.design_snr} dB')
print()
print('\tSNR (dB)|\tBER\t|\tFER')

for snr in snr_range:
    ber = 0
    fer = 0

    for _ in range(messages):
        msg = generate_binary_message(size=K)
        encoded = codec.encode(msg)
        transmitted = bpsk.transmit(message=encoded, snr_db=snr)
        decoded = codec.decode(transmitted)

        bit_errors, frame_error = compute_fails(msg, decoded)
        ber += bit_errors
        fer += frame_error

    result_ber[snr] = ber / (messages * codec.K)
    result_fer[snr] = fer / messages

    print(f'\t{snr}\t|\t{result_ber[snr]:.6f}\t|\t{result_fer[snr]:.6f}')