import logging

import numpy as np
import numpy.typing as npt

import global_config

config = global_config.Config()

from factor_mat import FactorMat
from nnf import NNF

class BinarySearchNNF:

    def __init__(self, nnf: NNF, idx: int, non_feature_p_sep: frozenset[int],
                 child_ordered_features: tuple[int, ...],
                 var_n_ary: np.ndarray, var_names: tuple[str, ...]) -> None:
        self.nnf = nnf
        self.idx = idx
        self.y_idx = len(var_n_ary) - 1
        self.non_feature_p_sep = non_feature_p_sep

        self.sorted_ordered_features: tuple[int, ...] = tuple(
            sorted(child_ordered_features))

        self.n_features = len(self.sorted_ordered_features)
        self.expected_fac_vars = tuple(
            sorted(
                tuple(self.non_feature_p_sep) + self.sorted_ordered_features))
        self.var_names = var_names

        self._Z_size = int(
            np.multiply.reduce(var_n_ary[list(self.non_feature_p_sep)]))

        self.features_nary = tuple(var_n_ary[list(
            self.sorted_ordered_features)])
        self.n_total_tests = int(np.multiply.reduce(self.features_nary))

        self._expected_mat_size = np.multiply.reduce(var_n_ary[list(
            self.expected_fac_vars)])
        assert self._expected_mat_size == self._Z_size * self.n_total_tests

        self.readonly_v_pvz: FactorMat | None = None
        self.arr_idx_to_nnf_idx: npt.NDArray[np.int_]

    def set_v_pvz(self, fac: FactorMat, skip_check=False) -> None:
        if not skip_check:
            assert self.expected_fac_vars == fac.vs, (self.expected_fac_vars,
                                                      fac.vs)
            assert fac.mat.size == self._expected_mat_size
            assert np.all(fac.mat >= 0)
        self.readonly_v_pvz = fac

    @property
    def is_cache_filled(self) -> bool:
        return self.readonly_v_pvz is not None

