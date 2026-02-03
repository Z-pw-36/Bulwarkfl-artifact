import torch
import copy
from collections import OrderedDict
import pickle
import os
import logging
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from ..common.ml_engine_backend import MLEngineBackend


class FLFHE:
    _fhe_instance = None

    @staticmethod
    def get_instance():
        if FLFHE._fhe_instance is None:
            FLFHE._fhe_instance = FLFHE()
        return FLFHE._fhe_instance

    def __init__(self):
        self.context_file = None
        self.he_context = None
        self.is_enabled = False

        # Threshold-HE (ThLHE) simulation controls
        self.fhe_mode: str = "single"
        self.enable_thlhe: bool = False
        self.thlhe_n: int = 0
        self.thlhe_t: int = 0
        self._thlhe_bundle: Optional[Dict[str, Any]] = None
        self.total_client_number: int = 1

    def is_fhe_enabled(self):
        return self.is_enabled

    def is_threshold_enabled(self) -> bool:
        return bool(self.enable_thlhe)

    # -------------------------
    # Utilities for bundle mode
    # -------------------------
    @staticmethod
    def _kdf_stream(key: bytes, length: int) -> bytes:
        out = b""
        counter = 0
        while len(out) < length:
            out += hashlib.sha256(key + counter.to_bytes(4, "big")).digest()
            counter += 1
        return out[:length]

    @classmethod
    def _xor_wrap(cls, data: bytes, key: bytes) -> bytes:
        stream = cls._kdf_stream(key, len(data))
        return bytes(a ^ b for a, b in zip(data, stream))

    @staticmethod
    def _prime_521() -> int:
        return (1 << 521) - 1

    @staticmethod
    def _shamir_combine(shares: List[Tuple[int, int]], prime: int) -> int:
        # Lagrange interpolation at x=0
        def inv(a: int) -> int:
            return pow(a, prime - 2, prime)

        secret = 0
        for j, (xj, yj) in enumerate(shares):
            num = 1
            den = 1
            for m, (xm, _) in enumerate(shares):
                if m == j:
                    continue
                num = (num * (-xm)) % prime
                den = (den * (xj - xm)) % prime
            lj = (num * inv(den)) % prime
            secret = (secret + yj * lj) % prime
        return secret

    def _load_bundle(self, obj: Any) -> Dict[str, Any]:
        # Legacy context.pickle stores raw bytes; new bundle stores a dict
        if isinstance(obj, dict) and obj.get("format") == "thlhe_bundle_v1":
            return obj
        if isinstance(obj, (bytes, bytearray)):
            # Wrap legacy bytes into a minimal bundle
            return {
                "format": "thlhe_bundle_v1",
                "he_lib": "tenseal",
                "n": 0,
                "t": 0,
                "prime_bits": 521,
                "wrap_key_len": 32,
                "wrapped_secret_ctx": None,
                "shares": None,
                "public_context_bytes": None,
                "legacy_secret_context_bytes": bytes(obj),
            }
        raise TypeError("Unsupported context.pickle content type: %s" % type(obj))

    def _unwrap_secret_context_bytes(self, shares: Optional[List[Tuple[int, int]]] = None) -> bytes:
        if self._thlhe_bundle is None:
            raise RuntimeError("ThLHE bundle not initialized")

        wrapped = self._thlhe_bundle.get("wrapped_secret_ctx", None)
        legacy = self._thlhe_bundle.get("legacy_secret_context_bytes", None)

        # If wrapped not available, fall back to legacy secret context bytes
        if wrapped is None:
            if isinstance(legacy, (bytes, bytearray)):
                return bytes(legacy)
            raise RuntimeError("No secret context bytes available for decryption")

        prime = self._prime_521()
        all_shares = self._thlhe_bundle.get("shares") or []
        t = int(self._thlhe_bundle.get("t") or self.thlhe_t or 2)

        if shares is None:
            shares = all_shares[:t]

        if shares is None or len(shares) < t:
            raise RuntimeError("Not enough shares to reconstruct wrapping key (need %d)" % t)

        wrap_key_int = self._shamir_combine(shares[:t], prime)
        wrap_key_len = int(self._thlhe_bundle.get("wrap_key_len") or 32)
        wrap_key = wrap_key_int.to_bytes(wrap_key_len, "big", signed=False)

        return self._xor_wrap(wrapped, wrap_key)

    # -------------------------
    # Main init
    # -------------------------
    def init(self, args):
        # Decide enablement from args
        self.is_enabled = bool(getattr(args, "enable_fhe", False))
        if not self.is_enabled:
            return

        import tenseal as fhe_core

        logging.info(".......init homomorphic encryption.......")
        self.total_client_number = int(getattr(args, "client_num_in_total", 1))

        self.fhe_mode = str(getattr(args, "fhe_mode", "single")).lower()
        self.enable_thlhe = bool(getattr(args, "enable_thlhe", False)) or self.fhe_mode in ("threshold", "thlhe")
        self.thlhe_t = int(getattr(args, "thlhe_threshold", 2))
        self.thlhe_n = int(getattr(args, "thlhe_n", self.total_client_number))

        # read in he context file
        ctx_path = os.path.join(os.path.dirname(__file__), "context.pickle")
        with open(ctx_path, "rb") as handle:
            raw_obj = pickle.load(handle)

        self._thlhe_bundle = self._load_bundle(raw_obj)

        # Load a context used for encryption / homomorphic ops.
        # In ThLHE mode we prefer a public context (no secret key) if present.
        secret_bytes = self._thlhe_bundle.get("legacy_secret_context_bytes", None)
        pub_bytes = self._thlhe_bundle.get("public_context_bytes", None)

        if self.enable_thlhe:
            # Derive public context bytes if absent (best-effort)
            if pub_bytes is None and isinstance(secret_bytes, (bytes, bytearray)):
                try:
                    sec_ctx = fhe_core.context_from(secret_bytes)
                    try:
                        pub_bytes = sec_ctx.serialize(save_secret_key=False)
                    except TypeError:
                        pub_bytes = sec_ctx.serialize(False)
                    self._thlhe_bundle["public_context_bytes"] = pub_bytes
                except Exception as e:
                    logging.warning("Unable to derive public context bytes, falling back to secret context for ops: %s", e)
                    pub_bytes = secret_bytes

            if not isinstance(pub_bytes, (bytes, bytearray)):
                raise RuntimeError("ThLHE enabled but no usable context bytes found")

            self.context_file = pub_bytes
            self.he_context = fhe_core.context_from(self.context_file)
        else:
            # Single-key mode (legacy): context bytes include secret key
            if isinstance(raw_obj, (bytes, bytearray)):
                self.context_file = bytes(raw_obj)
            else:
                # if bundle, use legacy secret bytes
                self.context_file = bytes(secret_bytes)
            self.he_context = fhe_core.context_from(self.context_file)

        # Disable for unsupported engines
        if hasattr(args, MLEngineBackend.ml_engine_args_flag) and args.ml_engine in [
            MLEngineBackend.ml_engine_backend_tf,
            MLEngineBackend.ml_engine_backend_jax,
            MLEngineBackend.ml_engine_backend_mxnet,
        ]:
            logging.info(
                "FL-HE is not supported for the machine learning engine: %s. "
                "We will support more engines in the future iteration." % args.ml_engine
            )
            self.is_enabled = False

    # -------------------------
    # Encrypt / aggregate / decrypt
    # -------------------------
    def fhe_enc(self, enc_type, model_params):
        import tenseal as fhe_core

        # transform tensor to encrypted form
        weight_factors = copy.deepcopy(model_params)
        for key in weight_factors.keys():
            weight_factors[key] = torch.flatten(torch.full_like(weight_factors[key], 1 / self.total_client_number))

        if enc_type == "local":
            np_params = OrderedDict()
            for key in model_params.keys():
                prepared_tensor = (torch.flatten(model_params[key])) * weight_factors[key]
                np_params[key] = fhe_core.plain_tensor(prepared_tensor)

            enc_model_params = OrderedDict()
            for key in np_params.keys():
                enc_model_params[key] = (fhe_core.ckks_vector(self.he_context, np_params[key])).serialize()
            return enc_model_params
        else:
            # not supported in the current version (kept for compatibility)
            enc_raw_client_model_or_grad_list = []
            for i in range(len(model_params)):
                local_sample_number, local_model_params = model_params[i]
                np_params = OrderedDict()
                for key in local_model_params.keys():
                    np_params[key] = torch.flatten(local_model_params[key]).numpy()

                enc_model_params = OrderedDict()
                for key in np_params.keys():
                    # Placeholder: original code expects self.fhe_helper in some builds
                    enc_model_params[key] = np_params[key]
                enc_raw_client_model_or_grad_list.append((local_sample_number, enc_model_params))
            return enc_raw_client_model_or_grad_list

    def fhe_fedavg(self, list_enc_model_parmas):
        import tenseal as fhe_core

        # init a template model
        n_clients = len(list_enc_model_parmas)
        _, temp_model_params = list_enc_model_parmas[0]
        enc_global_params = copy.deepcopy(temp_model_params)

        for i in range(n_clients):
            list_enc_model_parmas[i] = list_enc_model_parmas[i][1]
            for key in enc_global_params.keys():
                list_enc_model_parmas[i][key] = fhe_core.ckks_vector_from(self.he_context, list_enc_model_parmas[i][key])

        for key in enc_global_params.keys():
            for i in range(n_clients):
                if i != 0:
                    temp = list_enc_model_parmas[i][key]
                    list_enc_model_parmas[0][key] = list_enc_model_parmas[0][key] + temp

        for key in enc_global_params.keys():
            list_enc_model_parmas[0][key] = list_enc_model_parmas[0][key].serialize()

        enc_global_params = list_enc_model_parmas[0]
        return enc_global_params

    def fhe_dec(self, template_model_params, enc_model_params, shares: Optional[List[Tuple[int, int]]] = None):
        import tenseal as fhe_core

        # In single-key mode, self.he_context includes the secret key -> decrypt directly.
        # In ThLHE mode, decrypt uses a reconstructed secret context (best-effort simulation).
        dec_context = self.he_context
        if self.enable_thlhe:
            secret_bytes = self._unwrap_secret_context_bytes(shares=shares)
            dec_context = fhe_core.context_from(secret_bytes)

        params_shape = OrderedDict()
        for key in template_model_params.keys():
            params_shape[key] = template_model_params[key].size()

        params_tensor = OrderedDict()
        for key in enc_model_params.keys():
            enc_vec = fhe_core.ckks_vector_from(dec_context, enc_model_params[key])
            params_tensor[key] = torch.FloatTensor(enc_vec.decrypt())

        for key in params_tensor.keys():
            params_tensor[key] = torch.reshape(params_tensor[key], tuple(list(params_shape[key])))
        return params_tensor
