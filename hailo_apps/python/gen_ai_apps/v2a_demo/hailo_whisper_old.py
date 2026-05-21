import numpy as np
import hailo_platform as hpf

# try:
#     import hailo_platform as hpf  # pyHailoRT
# except Exception as e:
#     raise RuntimeError(
#         "pyHailoRT not found. Install HailoRT and ensure the Python bindings are on PYTHONPATH."
#     ) from e
DEBUG_HAILO = True

def _np_c(a):
    """Make C-contiguous float32 array."""
    return np.require(a, dtype=np.float32, requirements=["C"])

class HailoWhisperEncoder:
    """
    Run Whisper encoder on Hailo via VStreams.
    Expects mel features shaped like (1, 80, T_window). Returns encoder states [1, T_enc, D].
    """

    def __init__(
        self,
        hef_path: str,
        float_output: bool = True,
        interface: "hpf.HailoStreamInterface" = None,
        scheduler: str = "NONE",
    ):
        self.hef_path = hef_path
        self.hef = hpf.HEF(hef_path)

        # Create device with chosen scheduling policy (NONE is simplest; MPS if you need sharing)
        vdev_params = hpf.VDevice.create_params()
        sched_map = {
            "NONE": hpf.HailoSchedulingAlgorithm.NONE,
            "ROUND_ROBIN": hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
        }
        vdev_params.scheduling_algorithm = sched_map.get(
            str(scheduler).upper(), hpf.HailoSchedulingAlgorithm.NONE
        )
        self.device = hpf.VDevice(params=vdev_params)

        if interface is None:
            interface = hpf.HailoStreamInterface.PCIe

        # Configure network group from HEF
        cfg_params = hpf.ConfigureParams.create_from_hef(self.hef, interface=interface)
        self.network_group = self.device.configure(self.hef, cfg_params)[0]
        self.network_group_params = self.network_group.create_params()

        # Get stream infos
        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_info = self.hef.get_output_vstream_infos()[0]

        # VStream params (use float32 on both ends when float_output=True)
        fmt = hpf.FormatType.FLOAT32 if float_output else hpf.FormatType.AUTO
        self.in_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=fmt
        )
        self.out_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=fmt
        )

        print(f"[HailoWhisperEncoder] Input stream:  {self.in_info.name},  shape={tuple(self.in_info.shape)}, dtype=float32")
        print(f"[HailoWhisperEncoder] Output stream: {self.out_info.name}, shape={tuple(self.out_info.shape)}, dtype=float32")

    # def _prepare_input_frame(self, mel_1x80xT: np.ndarray) -> np.ndarray:
    #     """
    #     Align host mel features to HEF input frame shape (without batch).
    #     Common HEF shapes are (80, T) or (80, T, 1). We try the typical transposes.
    #     """
    #     if mel_1x80xT.dtype != np.float32:
    #         mel_1x80xT = mel_1x80xT.astype(np.float32, copy=False)

    #     x = mel_1x80xT
    #     if x.ndim != 3 or x.shape[0] != 1:
    #         raise ValueError(f"Expected mel features of shape (1,80,T). Got {x.shape}.")
    #     x = x[0]  # now (80, T)

    #     fshape = tuple(self.in_info.shape)

    #     # Case A: (80, T)
    #     if len(fshape) == 2:
    #         if x.shape == fshape:
    #             frame = x
    #         elif x.T.shape == fshape:
    #             frame = x.T
    #         else:
    #             raise ValueError(f"Mel shape {x.shape} does not match HEF input {fshape}.")
    #     # Case B: (80, T, 1)
    #     elif len(fshape) == 3 and fshape[-1] == 1:
    #         hw = fshape[:2]
    #         if x.shape == hw:
    #             frame = np.expand_dims(x, -1)
    #         elif x.T.shape == hw:
    #             frame = np.expand_dims(x.T, -1)
    #         else:
    #             raise ValueError(f"Mel shape {x.shape} does not match HEF input {fshape}.")
    #     else:
    #         raise ValueError(f"Unsupported HEF input shape {fshape} for Whisper encoder.")

    #     # Add batch dimension for the VStream API -> (1, ...) as required
    #     return np.expand_dims(frame, axis=0).astype(np.float32, copy=False)
    def _prepare_input_frame(self, mel_1x80xT: np.ndarray) -> np.ndarray:
        """
        Accept (1,80,T) float32 and return an array whose shape is EXACTLY
        self.in_info.shape, with C-contiguous memory.
        Handles common layouts: (80,T), (T,80), (80,T,1), (T,80,1), (1,80,T),
        (1,T,80), (1,80,T,1), (1,T,80,1). Never adds extra batch unless target has it.
        """
        if mel_1x80xT.ndim != 3 or mel_1x80xT.shape[0] != 1:
            raise ValueError(f"Expected mel shape (1,80,T), got {mel_1x80xT.shape}")
        if mel_1x80xT.dtype != np.float32:
            mel_1x80xT = mel_1x80xT.astype(np.float32, copy=False)

        target = tuple(self.in_info.shape)
        x80T = mel_1x80xT[0]                  # (80, T)
        T = x80T.shape[1]

        # Candidate tensors (no extra expand unless needed)
        candidates = [
            x80T,              # (80,T)
            x80T.T,            # (T,80)
            x80T[..., None],   # (80,T,1)
            x80T.T[..., None], # (T,80,1)
            mel_1x80xT,                     # (1,80,T)
            np.transpose(mel_1x80xT, (0,2,1)),         # (1,T,80)
            mel_1x80xT[..., None],                     # (1,80,T,1)
            np.transpose(mel_1x80xT, (0,2,1))[..., None],  # (1,T,80,1)
        ]

        for c in candidates:
            if c.shape == target:
                out = _np_c(c)
                if DEBUG_HAILO:
                    exp = int(np.prod(target) * 4)
                    print(f"[HailoWhisperEncoder] Prepared frame shape={out.shape}, nbytes={out.nbytes} (exp {exp})")
                return out

        # Special-case common HEF shapes we�ve seen:
        # (1, T, 80) where T matches our T
        if len(target) == 3 and target[0] == 1 and target[1] == T and target[2] == 80:
            out = _np_c(np.transpose(mel_1x80xT, (0,2,1)))  # (1,T,80)
            if DEBUG_HAILO:
                exp = int(np.prod(target) * 4)
                print(f"[HailoWhisperEncoder] Prepared frame shape={out.shape}, nbytes={out.nbytes} (exp {exp})")
            return out

        raise ValueError(f"Cannot map mel (1,80,{T}) to HEF input {target}")


    def encode(self, mel_1x80xT: np.ndarray) -> np.ndarray:
        frame = self._prepare_input_frame(mel_1x80xT)  # EXACT vstream shape
        # Belt & suspenders: verify byte size matches what Hailo expects
        expected_nbytes = int(np.prod(self.in_info.shape) * 4)  # float32
        if frame.nbytes != expected_nbytes:
            raise ValueError(f"Bad input size: got {frame.nbytes} bytes, expected {expected_nbytes} for {self.in_info.shape}")

        with self.network_group.activate(self.network_group_params):
            with hpf.InferVStreams(self.network_group, self.in_params, self.out_params) as pipe:
                outputs = pipe.infer({self.in_info.name: frame})

        enc = outputs[self.out_info.name]
        if enc.ndim == 2:
            enc = np.expand_dims(enc, 0)  # [1, T_enc, D] for HF decoder
        return _np_c(enc)


    def close(self):
        try:
            self.device.release()
        except Exception:
            pass