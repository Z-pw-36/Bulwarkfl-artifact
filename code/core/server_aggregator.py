import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple
from .. import Context
from ..contribution.contribution_assessor_manager import ContributionAssessorManager
from ..dp.FL_differential_privacy import FLDifferentialPrivacy
from ..security.FL_attacker import FLAttacker
from ..security.FL_defender import FLDefender
from ...ml.aggregator.agg_operator import FLAggOperator
from ..fhe.fhe_agg import FLFHE


class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args
        FLAttacker.get_instance().init(args)
        FLDefender.get_instance().init(args)
        FLDifferentialPrivacy.get_instance().init(args)
        FLFHE.get_instance().init(args)
        self.contribution_assessor_mgr = ContributionAssessorManager(args)
        self.final_contribution_assigment_dict = dict()

        self.eval_data = None

    def is_main_process(self):
        return True

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def on_before_aggregation(
            self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]
    ):
        if FLFHE.get_instance().is_fhe_enabled():
            logging.info(" ---- loading encrypted models ----")
            enc_raw_client_model_or_grad_list = raw_client_model_or_grad_list
            client_idxs = [i for i in range(len(raw_client_model_or_grad_list))]
            return enc_raw_client_model_or_grad_list, client_idxs
        else:
            if FLDifferentialPrivacy.get_instance().is_global_dp_enabled() and FLDifferentialPrivacy.get_instance().is_clipping():
                raw_client_model_or_grad_list = FLDifferentialPrivacy.get_instance().global_clip(raw_client_model_or_grad_list)
            if FLAttacker.get_instance().is_data_reconstruction_attack():
                FLAttacker.get_instance().reconstruct_data(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    extra_auxiliary_info=self.get_model_params(),
                )
            if FLAttacker.get_instance().is_model_attack():
                raw_client_model_or_grad_list = FLAttacker.get_instance().attack_model(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    extra_auxiliary_info=self.get_model_params(),
                )
            client_idxs = [i for i in range(len(raw_client_model_or_grad_list))]
            if FLDefender.get_instance().is_defense_enabled():
                raw_client_model_or_grad_list = FLDefender.get_instance().defend_before_aggregation(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    extra_auxiliary_info=self.get_model_params(),
                )
                client_idxs = FLDefender.get_instance().get_benign_client_idxs(client_idxs=client_idxs)

            return raw_client_model_or_grad_list, client_idxs

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        if FLFHE.get_instance().is_fhe_enabled():
            logging.info(" ---- aggregating models using homomorphic encryption ----")
            return FLFHE.get_instance().fhe_fedavg(raw_client_model_or_grad_list)
        else:
            if FLDefender.get_instance().is_defense_enabled():
                return FLDefender.get_instance().defend_on_aggregation(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    base_aggregation_func=FLAggOperator.agg,
                    extra_auxiliary_info=self.get_model_params(),
                )
            if FLDifferentialPrivacy.get_instance().to_compute_params_in_aggregation_enabled():
                FLDifferentialPrivacy.get_instance().set_params_for_dp(raw_client_model_or_grad_list)
            return FLAggOperator.agg(self.args, raw_client_model_or_grad_list)

    def on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        fhe = FLFHE.get_instance()
        if fhe.is_fhe_enabled():
            # Threshold mode: decrypt on server side (t-of-n gate simulated via shares in context bundle)
            if fhe.is_threshold_enabled():
                logging.info(" ---- threshold decrypting aggregated global model ----")
                template_model_params = self.get_model_params()
                return fhe.fhe_dec(template_model_params, aggregated_model_or_grad)

            logging.info(" ---- finish aggregating encrypted global model ----")
            return aggregated_model_or_grad
        else:
            if FLDifferentialPrivacy.get_instance().is_global_dp_enabled():
                logging.info("-----add central DP noise ----")
                aggregated_model_or_grad = FLDifferentialPrivacy.get_instance().add_global_noise(
                    aggregated_model_or_grad
                )
            if FLDefender.get_instance().is_defense_enabled():
                aggregated_model_or_grad = FLDefender.get_instance().defend_after_aggregation(aggregated_model_or_grad)
            return aggregated_model_or_grad

    def assess_contribution(self):
        if self.contribution_assessor_mgr is None:
            return
        # TODO: start to run contribution assessment in an independent python process
        client_num_per_round = len(Context().get(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND))
        client_index_for_this_round = Context().get(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND)
        local_weights_from_clients = Context().get(Context.KEY_CLIENT_MODEL_LIST)

        metric_results_in_the_last_round = Context().get(Context.KEY_METRICS_ON_LAST_ROUND)
        (acc_on_last_round, _, _, _) = metric_results_in_the_last_round
        metric_results_on_aggregated_model = Context().get(Context.KEY_METRICS_ON_AGGREGATED_MODEL)
        (acc_on_aggregated_model, _, _, _) = metric_results_on_aggregated_model
        test_data = Context().get(Context.KEY_TEST_DATA)
        validation_func = self.test
        self.contribution_assessor_mgr.run(
            client_num_per_round,
            client_index_for_this_round,
            FLAggOperator.agg,
            local_weights_from_clients,
            acc_on_last_round,
            acc_on_aggregated_model,
            test_data,
            validation_func,
            self.args.device,
        )

        if self.args.round_idx == self.args.comm_round - 1:
            self.final_contribution_assigment_dict = self.contribution_assessor_mgr.get_final_contribution_assignment()
            logging.info(
                "self.final_contribution_assigment_dict = {}".format(self.final_contribution_assigment_dict))

    @abstractmethod
    def test(self, test_data, device, args):
        pass

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        pass
