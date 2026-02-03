import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..security.FL_attacker import FLAttacker
from ...core.dp.FL_differential_privacy import FLDifferentialPrivacy
from ..fhe.fhe_agg import FLFHE


class ClientTrainer(ABC):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class can be used in both server and client side
    3. This class is an operator which does not cache any states inside.
    """

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args
        self.local_train_dataset = None
        self.local_test_dataset = None
        self.local_sample_number = 0
        self.rid = 0
        self.template_model_params: Optional[Any] = None
        self.enc_model_params = None
        FLDifferentialPrivacy.get_instance().init(args)
        FLAttacker.get_instance().init(args)
        FLFHE.get_instance().init(args)

    def set_id(self, trainer_id):
        self.id = trainer_id

    def is_main_process(self):
        return True

    def update_dataset(self, local_train_dataset, local_test_dataset, local_sample_number):
        if FLAttacker.get_instance().is_data_poisoning_attack() and FLAttacker.get_instance().is_to_poison_data():
            self.local_train_dataset = FLAttacker.get_instance().poison_data(local_train_dataset)
            self.local_test_dataset = FLAttacker.get_instance().poison_data(local_test_dataset)
        else:
            self.local_train_dataset = local_train_dataset
            self.local_test_dataset = local_test_dataset
        self.local_sample_number = local_sample_number

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def get_enc_model_params(self):
        return self.enc_model_params

    def set_enc_model_params(self, enc_model_parameters):
        self.enc_model_params = enc_model_parameters

    def on_before_local_training(self, train_data, device, args):
        fhe = FLFHE.get_instance()
        if fhe.is_fhe_enabled():
            if self.rid != 0:
                # Threshold mode: server side returns a plaintext global model.
                if fhe.is_threshold_enabled():
                    self.set_model_params(self.get_enc_model_params())
                else:
                    if self.template_model_params is None:
                        self.template_model_params = self.get_model_params()

                    # get encrypted global params, and decrypt then set params
                    logging.info(" ---- decrypting aggregated model ----")
                    dec_aggregated_model = fhe.fhe_dec(
                        self.template_model_params,
                        self.get_enc_model_params()
                    )
                    self.set_model_params(dec_aggregated_model)
            self.rid += 1

    @abstractmethod
    def train(self, train_data, device, args):
        pass

    def on_after_local_training(self, train_data, device, args):
        if FLFHE.get_instance().is_fhe_enabled():
            # encrypt before sending to server
            logging.info(" ---- encrypting client model ----")
            encrypted_model_params = FLFHE.get_instance().fhe_enc('local', self.get_model_params())
            self.set_enc_model_params(encrypted_model_params)
        elif FLDifferentialPrivacy.get_instance().is_local_dp_enabled():
            logging.info("-----add local DP noise ----")
            model_params_with_dp_noise = FLDifferentialPrivacy.get_instance().add_local_noise(
                self.get_model_params()
            )
            self.set_model_params(model_params_with_dp_noise)

    def test(self, test_data, device, args):
        pass
