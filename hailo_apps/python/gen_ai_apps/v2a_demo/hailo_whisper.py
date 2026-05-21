import numpy as np

try:
    import hailo_platform as hpf  # pyHailoRT
except Exception as e:
    raise RuntimeError(
        "pyHailoRT not found. Install HailoRT and ensure the Python bindings are on PYTHONPATH."
    ) from e


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

        # Interface: Raspberry Pi 5 AI HAT and PCIe cards use PCIe
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

        # Helpful prints once, so you can confirm shapes without digging into HEF
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

    #     fshape = tuple(self.in_info.shape)  # HEF frame shape (no batch)

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
        Accepts (1,80,T). HEF may want any of:
        (80,T), (T,80), (80,T,1), (1,80,T), (1,T,80), (1,80,T,1)
        We reshape/transposes accordingly and always return (batch=1, ...) for VStreams.
        """
        x = mel_1x80xT
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected mel shape (1,80,T), got {x.shape}.")

        # drop batch for matching
        x = x[0]  # (80, T)
        T = x.shape[1]
        fshape = tuple(self.in_info.shape)

        def add_batch(arr):
            return np.expand_dims(arr, 0).astype(np.float32, copy=False)

        # --- 2D (no batch/channel in HEF) ---
        if len(fshape) == 2:
            # (80, T)
            if fshape == (80, T):
                return add_batch(x)
            # (T, 80)
            if fshape == (T, 80):
                return add_batch(x.T)

        # --- 3D (one of dims might be batch=1 or channel=1) ---
        if len(fshape) == 3:
            # (1, 80, T)  -> exactly our host layout w/ batch in HEF
            if fshape == (1, 80, T):
                return add_batch(x)  # will become (1,80,T) with our added batch -> (1,80,T) OK
            # (1, T, 80)
            if fshape == (1, T, 80):
                return add_batch(x.T)
            # (80, T, 1)
            if fshape == (80, T, 1):
                return add_batch(np.expand_dims(x, -1))
            # (T, 80, 1)
            if fshape == (T, 80, 1):
                return add_batch(np.expand_dims(x.T, -1))
            # (1, 80, T,?) � Some builds report batch inside shape (you saw (1,500,80))
            # Handle (1, 500, 80) = (batch, T, 80)
            if fshape[0] == 1 and fshape[1:] == (T, 80):
                return add_batch(x.T)
            # Handle (1, 80, 500)
            if fshape[0] == 1 and fshape[1:] == (80, T):
                return add_batch(x)

        # --- 4D (batch + H/W + C) variants ---
        if len(fshape) == 4:
            # (1, 80, T, 1)
            if fshape == (1, 80, T, 1):
                return add_batch(np.expand_dims(x, -1))
            # (1, T, 80, 1)
            if fshape == (1, T, 80, 1):
                return add_batch(np.expand_dims(x.T, -1))

        raise ValueError(f"Mel (80,{T}) cannot be arranged to HEF input {fshape}")


    def encode(self, mel_1x80xT: np.ndarray) -> np.ndarray:
        """
        Run a single encoder pass.
        Returns: np.ndarray with shape [1, T_enc, D] (float32).
        """
        # Prepare data to match HEF input
        frame = self._prepare_input_frame(mel_1x80xT)

        # Activate and infer (open/close per call for simplicity)
        with self.network_group.activate(self.network_group_params):
            with hpf.InferVStreams(self.network_group, self.in_params, self.out_params) as pipe:
                outputs = pipe.infer({self.in_info.name: frame})

        enc = outputs[self.out_info.name]
        # Ensure [1, T_enc, D] for huggingface decoder
        if enc.ndim == 2:
            enc = np.expand_dims(enc, axis=0)
        return enc.astype(np.float32, copy=False)

    def close(self):
        try:
            self.device.release()
        except Exception:
            pass