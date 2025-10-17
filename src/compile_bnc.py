# %%
import logging
from datetime import datetime

_logger = logging.getLogger("compile_bnc")
import sys, json,  collections, time,  itertools
import typing, cProfile, pstats, io

import numpy as np
import numpy.typing as npt

import global_config

config = global_config.Config()

import utils
from factor_mat import FactorMat, FactorMatTrue
from nnf import NNF
from compiled_nnf_search import BinarySearchNNF
from joint_tree import JointTree, JointTreeNode
from arithmetic_circuit import ArithmeticCircuit

np.set_printoptions(precision=3)


class BNCCompiler:

    def __init__(self, network_name: str, y: str, threshold: float,
                 features: list[str]) -> None:
        FactorMat.CHECK_SUM_OUT_FEATURE = False
        self.network_name = network_name
        self.bn = utils.read_hugin_network(network_name)
        self.features = frozenset(features)
        assert self.features <= self.bn.leaf_var_strs
        assert y in self.bn.root_var_strs
        self.y = y
        self.n_vars = len(self.bn.parents)
        self.y_idx = self.n_vars - 1
        FactorMat.N_VARS = self.n_vars
        FactorMat.N_FEAS = len(self.features)
        assert self.y in self.bn.root_var_strs
        self.y_binary = len(self.bn.states[self.y]) == 2
        if self.y_binary:
            self._prob_thresh: float = threshold
            self._odds_thresh: float = threshold / (1 - threshold)
        else:  # argmax
            self._prob_thresh, self._odds_thresh = -1.0, -1.0

        self.compiled_nnf_search_tree: dict[int, BinarySearchNNF] = {}

        self.n_features = len(self.features)
        _logger.info(
            f"N Vars: {self.n_vars}, Y: {self.y}, N Features: {self.n_features}"
        )
        self.ordered_var_names, self.var_name_to_idx = utils.get_problem_var_name_order(
            self.bn.all_var_strs, self.features, self.y)
        self.bn.reorder_indices_with_ordeded_vars(self.ordered_var_names)
        if config.DEBUG_LVL > 1:
            self.bn.graphviz_render(name=network_name, out_dir=config.OUT_DIR)
        self.var_nary = np.array(
            [len(self.bn.states[v]) for v in self.ordered_var_names],
            dtype=np.int32)
        self.var_nary.setflags(write=False)
        self.NON_TARGET_CLSES = tuple(i for i in range(self.var_nary[-1])
                                      if i != config.TARGET_CLS)
        # XU: the family of X, X is the first var
        self.family_var_names = tuple(
            (x, ) + tuple(self.bn.parents[x]) for x in self.ordered_var_names)
        self.family_var_indices = tuple(
            tuple(self.var_name_to_idx[x] for x in vs)
            for vs in self.family_var_names)
        self.n_features = len(self.features)
        assert len(self.family_var_names[-1]) == 1

        self._nodes_cache_pvz: set[int] = set()
        self._arr_idx_to_nnf_cache: dict[int, npt.NDArray[np.int_]] = {}


        self._node_pvz_enumerated: set[int] = set()
        if config.DEBUG_LVL > 0:
            self._debug_stat: dict[str, dict] = {
                'T': collections.defaultdict(int),
                'F': collections.defaultdict(int),
                'nnf': collections.defaultdict(int),
            }

    def _recur_init_sep_and_pvz(self, n: JointTreeNode, p_idx: int,
                                p_remained_sep: frozenset[int]) -> None:
        ordered_features = self.joint_tree.ordered_expected_features[n.idx]
        visit_order = self.joint_tree.ordered_downstream_neighbors[n.idx]
        if ordered_features[0]:
            assert len(visit_order) <= 1
        ordered_features_flat = tuple(itertools.chain(*ordered_features))
        features_to_visit = frozenset(ordered_features_flat)
        p_sep = frozenset() if p_idx < 0 else n.separators[p_idx]
        size_ok = True
        if (size_ok or (not visit_order)) and p_idx >= 0:
            # update upward sep, add all expected features
            self._nodes_cache_pvz.add(n.idx)
            self.compiled_nnf_search_tree[n.idx] = BinarySearchNNF(
                self.nnf, n.idx, p_sep, ordered_features_flat, self.var_nary,
                self.ordered_var_names)
        if p_idx >= 0:
            assert not (features_to_visit & n.separators[p_idx])
            n.separators[p_idx] |= features_to_visit
        else:
            n.separators[p_idx] = features_to_visit | {self.y_idx}

        # handle downward msg
        remained_sep_list = []
        if n.idx in self.joint_tree.downward_msg_nodes:
            if visit_order:
                all_features = frozenset([i for i in range(self.n_features)])
                prev_features = all_features - features_to_visit
                assert len(ordered_features) == len(visit_order) + 1
                downstream_seps = tuple(n.separators[neighbor.idx]
                                        for neighbor in visit_order)
                assert len(ordered_features) == len(downstream_seps) + 1
                extra_downward_XY = prev_features | frozenset(
                    ordered_features[0]) | {self.y_idx}
                for i, neighbor in enumerate(visit_order[:-1]):
                    remained_sep: frozenset[int] = frozenset(
                        itertools.chain.from_iterable(downstream_seps[i + 1:]))
                    remained_sep_list.append(remained_sep | p_remained_sep)
                    n.separators[neighbor.idx] |= (extra_downward_XY
                                                   | remained_sep
                                                   | p_remained_sep)
                    extra_downward_XY |= frozenset(ordered_features[i + 1])
                r_idx = visit_order[-1].idx
                n.separators[r_idx] |= (extra_downward_XY | p_remained_sep)
                remained_sep_list.append(p_remained_sep)
            else:  # leaf node
                assert frozenset(ordered_features_flat) < n.cluster

        if visit_order:
            for i, neighbor in enumerate(visit_order):
                new_p_remained_sep = p_remained_sep
                if remained_sep_list:
                    new_p_remained_sep |= remained_sep_list[i]
                self._recur_init_sep_and_pvz(neighbor, n.idx,
                                             new_p_remained_sep)

        else:  # leaf
            assert ordered_features[0]
            st = self.compiled_nnf_search_tree[n.idx]
            p = self.joint_tree.original_pvz[n.idx]
            st.set_v_pvz(p.project_to(n.separators[p_idx]))
            self._build_leaf_arr_idx_to_nnf_idx(n)
            _expected_shape = tuple(self.var_nary[list(
                ordered_features[0])].tolist())
            _actual_shape = self._arr_idx_to_nnf_cache[n.idx].shape
            assert _actual_shape == _expected_shape
            self._node_pvz_enumerated.add(n.idx)

    def compute_pvz_and_arr_idx_to_nnf(self, n: JointTreeNode,
                                       p_idx: int) -> None:
        st = self.compiled_nnf_search_tree[n.idx]
        if st.readonly_v_pvz is not None:
            assert n.idx in self._arr_idx_to_nnf_cache
            return
        assert n.idx not in self._arr_idx_to_nnf_cache

        visit_order = self.joint_tree.ordered_downstream_neighbors[n.idx]
        assert visit_order  # leaf node pvz is filled in init
        for c in visit_order:
            self.compute_pvz_and_arr_idx_to_nnf(c, n.idx)

        old_pvz = self.joint_tree.original_pvz[n.idx]
        p_sep = n.separators[p_idx]
        l, r = visit_order[0], visit_order[-1]
        pvz_r = self.compiled_nnf_search_tree[r.idx].readonly_v_pvz
        assert pvz_r is not None
        tmp = old_pvz.factor_mult(pvz_r)
        if l == r:
            pvz = tmp.project_to(p_sep)
            self.compiled_nnf_search_tree[n.idx].set_v_pvz(pvz)
            self._arr_idx_to_nnf_cache[n.idx] = self._arr_idx_to_nnf_cache[
                r.idx]
            return

        pvz_l = self.compiled_nnf_search_tree[l.idx].readonly_v_pvz
        assert pvz_l is not None
        pvz = tmp.factor_mult(pvz_l).project_to(p_sep)
        self.compiled_nnf_search_tree[n.idx].set_v_pvz(pvz)
        all_feas = self.joint_tree.ordered_expected_features[n.idx]
        sorted_feas = sorted(itertools.chain.from_iterable(all_feas))
        nary = tuple(self.var_nary[sorted_feas].tolist())
        l_feas, r_feas = all_feas[1], all_feas[2]
        l_axis = np.searchsorted(sorted_feas, l_feas)
        r_axis = np.searchsorted(sorted_feas, r_feas)
        d = np.empty(nary, dtype=np.int_)
        l_idx_to_nnf = self._arr_idx_to_nnf_cache[l.idx]
        r_idx_to_nnf = self._arr_idx_to_nnf_cache[r.idx]
        for idx in np.ndindex(nary):
            l_idx = tuple(idx[i] for i in l_axis)
            r_idx = tuple(idx[i] for i in r_axis)
            l_nnf = l_idx_to_nnf[l_idx].item()
            r_nnf = r_idx_to_nnf[r_idx].item()
            d[idx] = self.nnf.make_AND(l_nnf, r_nnf)
        d.setflags(write=False)
        self._arr_idx_to_nnf_cache[n.idx] = d

    def _build_leaf_arr_idx_to_nnf_idx(self, n: JointTreeNode) -> None:
        assert n.idx not in self._node_pvz_enumerated
        st = self.compiled_nnf_search_tree[n.idx]
        pvz = st.readonly_v_pvz
        assert pvz is not None
        feas = list(v for v in pvz.vs if (v < self.n_features))
        if config.DEBUG_LVL > 0:
            assert feas
            _expected_feas = self.joint_tree.ordered_expected_features[
                n.idx][0]
            assert tuple(feas) == _expected_feas
        nary = self.var_nary[feas]
        d = np.empty(nary, dtype=np.int_)

        all_lits = [[self.nnf.make_lit(x, j) for j in range(self.var_nary[x])]
                    for x in feas]
        for arr_idx in np.ndindex(*nary):
            lits = tuple(y[x] for x, y in zip(arr_idx, all_lits))
            d[arr_idx] = self.nnf.make_AND_binary_tree(lits)
        d.flags.writeable = False
        self._arr_idx_to_nnf_cache[n.idx] = d

    def _enumerate_pvz_from_cache(
            self,
            n: JointTreeNode) -> typing.Iterator[tuple[int, FactorMat, tuple]]:
        pvz = self.compiled_nnf_search_tree[n.idx].readonly_v_pvz
        assert pvz is not None

        d = self._arr_idx_to_nnf_cache[n.idx]
        for arr_idx, nnf_idx in np.ndenumerate(d):
            slice_idx = (slice(pos, pos + 1) for pos in arr_idx)
            p = FactorMat(mat=pvz.mat[*slice_idx],
                          vs=pvz.vs,
                          nary=pvz.nary,
                          var_names=pvz.var_names,
                          cond_vars=pvz.cond_vars)
            yield nnf_idx.item(), p, arr_idx

    def send_msg_and_compile(self, n: JointTreeNode, p_idx: int,
                             msg: FactorMat) -> list[int]:
        is_global_last = n.idx in self.joint_tree._global_last_nodes
        assert is_global_last
        visit_order = self.joint_tree.ordered_downstream_neighbors[n.idx]
        is_decide_node = n.idx in self.joint_tree.decide_nodes and (
            not config.DISABLE_DECIDE_NODE)
        if config.DEBUG_LVL > 0:
            assert n.idx in self.joint_tree.downward_msg_nodes
            if p_idx >= 0:
                assert msg.vs[-1] == self.y_idx

        if config.DEBUG_LVL > 0:
            if is_decide_node:
                neg_mask, pos_mask = self.decide_partial(msg, False)
                assert (not pos_mask) and (not neg_mask)

        new_fac = msg.factor_mult(
            self.joint_tree.original_cluster_joints[n.idx])

        if config.DEBUG_LVL > 0:
            assert len(visit_order) <= 2
            assert not new_fac.cond_vars

        nnf_idx = self.nnf.NNF_FALSE_IDX
        or_children_indices: list[int]
        if not visit_order:  # leaf node
            p_yxz = new_fac.mat
            if p_yxz.ndim > self.n_features + 1:
                p_yxz = np.sum(p_yxz,
                               axis=tuple(
                                   range(self.n_features, p_yxz.ndim - 1)))
            p_yxz = p_yxz.squeeze()
            if self.y_binary:
                neg_mask = (p_yxz[..., config.TARGET_CLS] -
                            p_yxz[..., 1 - config.TARGET_CLS]) <= 0
            else:
                inst_cls = np.argmax(p_yxz, axis=-1)
                neg_mask = inst_cls != config.TARGET_CLS
            arr_idx_to_nnf = self._arr_idx_to_nnf_cache[n.idx]
            or_children_indices = arr_idx_to_nnf[neg_mask].tolist()
        else:
            or_children_indices = []
            all_feas = self.joint_tree.ordered_expected_features[n.idx]
            l_feas = all_feas[1]
            l, r = visit_order[0], visit_order[-1]
            tmp = new_fac.factor_mult(
                self.joint_tree.downward_msg_div[n.idx][r.idx],
                divide=True,
                normalize=True)
            r_sep = n.separators[r.idx]
            r_is_decide = (r.idx in self.joint_tree.decide_nodes) and (
                not config.DISABLE_DECIDE_NODE)
            if l == r:
                new_msg = tmp.project_to(r_sep)
                if r_is_decide:
                    neg_mask, pos_mask = self.decide_partial(
                        new_msg, _debug_batch=False)
                    if neg_mask:
                        return [self.nnf.NNF_TRUE_IDX]
                    if pos_mask:
                        return []
                for r_inst in self.send_msg_and_compile(r, n.idx, new_msg):
                    assert r_inst > 0
                    if r_inst == self.nnf.NNF_TRUE_IDX:
                        return [self.nnf.NNF_TRUE_IDX]
                    or_children_indices.append(r_inst)

            else:
                self.compute_pvz_and_arr_idx_to_nnf(l, n.idx)
                l_pvz = self.compiled_nnf_search_tree[l.idx].readonly_v_pvz
                assert l_pvz is not None
                l_arr_idx_to_nnf = self._arr_idx_to_nnf_cache[l.idx]
                assert l_arr_idx_to_nnf.ndim == len(l_feas)
                r_msgs = tmp.factor_mult(l_pvz,
                                         normalize=True).project_to(r_sep)

                slicer = [slice(None) for _ in range(r_msgs.mat.ndim)]
                l_axes = np.searchsorted(r_msgs.vs, l_feas)
                if r_is_decide:
                    neg_mask, pos_mask = self.decide_partial(r_msgs, True)
                    non_final_l_mask = ~(neg_mask | pos_mask)
                    or_children_indices.extend(
                        l_arr_idx_to_nnf[neg_mask].tolist())
                    non_final_l_indices = np.argwhere(non_final_l_mask)
                else:
                    l_nary = tuple(self.var_nary[list(l_feas)].tolist())
                    non_final_l_indices = np.ndindex(l_nary)

                for arr_idx in non_final_l_indices:
                    for ax, idx in zip(l_axes, arr_idx):
                        slicer[ax] = slice(idx, idx + 1)
                    new_msg_arr = r_msgs.mat[*slicer]
                    new_msg = FactorMat(new_msg_arr, r_msgs.vs, r_msgs.nary,
                                        r_msgs.var_names, r_msgs.cond_vars)
                    l_inst = l_arr_idx_to_nnf[tuple(arr_idx)].item()
                    r_set: list[int] = []
                    for r_inst in self.send_msg_and_compile(r, n.idx, new_msg):
                        assert r_inst > 0
                        r_set.append(r_inst)
                        if r_inst == self.nnf.NNF_TRUE_IDX:
                            break
                    if r_set:
                        r_set_or = self.nnf.make_OR(frozenset(r_set))
                        lr_inst = self.nnf.make_AND(l_inst, r_set_or)
                        or_children_indices.append(lr_inst)

        if or_children_indices:
            assert n.idx in self.joint_tree._global_last_nodes
            nnf_idx = self.nnf.make_OR(frozenset(or_children_indices))
            or_children_indices = [nnf_idx]

        return or_children_indices

    def debug_or(self, or_children_indices: list[int]) -> int:
        assert config.DEBUG_LVL > 1
        nnf_idx = or_children_indices[0]
        for idx, i in enumerate(or_children_indices):
            if idx == 0:
                nnf_idx = i
                _mask = self.nnf.get_all_models(nnf_idx)
                continue
            _mask |= self.nnf.get_all_models(i)
            nnf_idx = self.nnf.make_OR(frozenset([i, nnf_idx]))
            _other_mask = self.nnf.get_all_models(nnf_idx)
            assert np.all(_mask == _other_mask), ([
                self.nnf.node_to_str(j) for j in or_children_indices[:idx]
            ], self.nnf.node_to_str(i), self.nnf.node_to_str(nnf_idx))
        assert nnf_idx > 0
        return nnf_idx

    def decide_partial(
        self, new_fac: FactorMat, _debug_batch: bool
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        # return neg inst mask, pos inst mask
        p_yua = new_fac.mat
        iter_axes = [
            i for i in range(p_yua.ndim)
            if new_fac.vs[i] < self.n_features and p_yua.shape[i] > 1
        ]
        m = len(iter_axes)
        iter_shape = tuple(p_yua.shape[i] for i in iter_axes)
        p_yua_reshaped = p_yua.squeeze().reshape(*iter_shape, -1,
                                                 self.var_nary[-1])
        target_p_yua = p_yua_reshaped[..., (config.TARGET_CLS, )]
        non_target_p_yua = p_yua_reshaped[..., self.NON_TARGET_CLSES]

        le_mask = np.all(target_p_yua <= non_target_p_yua, axis=-2)
        neg_inst_mask = np.any(le_mask, axis=-1)

        gt_mask = np.all(target_p_yua > non_target_p_yua, axis=-2)
        pos_inst_mask = np.all(gt_mask, axis=-1)

        if config.DEBUG_LVL > 0:
            assert _debug_batch == (m > 0)
            assert le_mask.shape == iter_shape + (len(self.NON_TARGET_CLSES), )
            assert _debug_batch == (len(neg_inst_mask.shape) > 0)

        return neg_inst_mask, pos_inst_mask

    def compile_bnc(self) -> dict:
        if config.DEBUG_LVL > 0:
            utils.cleanup_prev_run_pdf(self.network_name)
            self.ac: ArithmeticCircuit = ArithmeticCircuit(self.bn)

        start_time = time.perf_counter()

        self.joint_tree = JointTree.try_jointrees(self.family_var_indices,
                                                  self.n_features,
                                                  self.ordered_var_names,
                                                  self.bn, self.var_nary)
        self.joint_tree.prepare_for_bnc(self._odds_thresh)

        _tmp_time1 = time.perf_counter()
        _logger.info(f"joint tree total: {_tmp_time1 - start_time:.2f} sec")
        self.nnf = NNF(self.network_name,
                       tuple(self.var_nary[:self.n_features].tolist()),
                       self.joint_tree.get_fea_to_global_order())
        self._recur_init_sep_and_pvz(self.joint_tree.root, -1, frozenset())
        _tmp_time2 = time.perf_counter()
        _logger.info(f"init sep and pvz: {_tmp_time2 - _tmp_time1:.2f} sec")
        FactorMat.CHECK_SUM_OUT_FEATURE = True
        compile_ret = self.send_msg_and_compile(self.joint_tree.root, -1,
                                                FactorMatTrue)
        assert len(compile_ret) == 1
        root = compile_ret[0]
        runtime_dur = time.perf_counter() - start_time
        _logger.warning(f"FINISHED compilation: {runtime_dur:.2f} secs")
        summary = self._compilation_result(root, runtime_dur)
        _logger.info(self.joint_tree.get_fea_to_global_order())
        self._maybe_check_root_prob_nnf(root)
        return summary

    def _compilation_result(self, root: int, runtime_dur: float) -> dict:
        if root == self.nnf.NNF_FALSE_IDX:
            n_nnf_nodes, n_nnf_edges = 1, 0
        else:
            n_nnf_nodes, n_nnf_edges = self.nnf.get_graph_size(root)

        _logger.warning("\n".join([
            f"Num created nodes (possible some dead): {len(self.nnf._nnf_idx_to_node)}",
            f"Num nodes in NNF: {n_nnf_nodes}, num edges: {n_nnf_edges}"
        ]))

        if n_nnf_nodes < 500 and config.DEBUG_LVL > 0:
            nnf_graphviz = self.nnf.to_graphviz(root, self.ordered_var_names)
            nnf_graphviz.render(directory=config.OUT_DIR)

        result_summary = {
            "network": self.network_name,
            "y": self.y,
            "yi": config.TARGET_CLS,
            "threshold": self._prob_thresh,
            "time": round(runtime_dur, 3),
            "n_features": self.n_features,
            "n_nodes": n_nnf_nodes,
            "n_edges": n_nnf_edges,
            "width": self.joint_tree._original_width,
            "ft_width": self.joint_tree.quality
        }
        _logger.warning(json.dumps(result_summary))

        return result_summary

    def _maybe_check_root_prob_nnf(self, root: int) -> None:
        if config.DEBUG_CHECK_NNF:
            feature_nary = tuple(self.var_nary[:self.n_features].tolist())
            pos_inst_mask = self.nnf.get_all_models(root)
            n_pos_inst = np.sum(pos_inst_mask).item()
            n_possible_instances = np.multiply.reduce(feature_nary)
            if self.n_features <= config.DEBUG_CHECK_FEATURES_LIMIT:
                true_joint = self.ac.get_ground_truth_joint_prob(
                    self.y, self.features)

                _logger.warning(
                    f"Number of models: {n_pos_inst} out of {n_possible_instances} all possible instances"
                )
                self._debug_check_mask(true_joint, pos_inst_mask)
            else:
                n_free_features = config.DEBUG_CHECK_FEATURES_LIMIT - 3
                for _ in range(8):
                    fac, tuple_idx = self.ac.sample_eval_joint(
                        self.y, self.features, n_free_features)
                    sub_mask = pos_inst_mask[tuple_idx]
                    self._debug_check_mask(fac.mat, sub_mask)
            _logger.info(f"NNF correct!")

    def _debug_check_mask(self, joint_prob: np.ndarray,
                          inst_mask: np.ndarray) -> None:
        if self.y_binary:
            likelihood = joint_prob[..., config.TARGET_CLS] / (
                joint_prob[..., config.TARGET_CLS] +
                joint_prob[..., 1 - config.TARGET_CLS])
            assert likelihood.shape == inst_mask.shape
            expected_mask = likelihood <= self._prob_thresh
            diff_mask = np.logical_xor(expected_mask, inst_mask)
            selected = likelihood[diff_mask]
            if selected.size > 0:
                assert np.allclose(selected.ravel(),
                                   self._prob_thresh), "WRONG MASK!!"
        else:
            expected_mask = np.argmax(joint_prob, axis=-1) != config.TARGET_CLS
            diff_mask = np.logical_xor(expected_mask, inst_mask)
            assert not np.any(diff_mask), "WRONG MASK!!"


def main():
    p = "andes-*-19.json"
    summary_list = []
    for _ in range(1):
        for fpath in config.PROBLEMS_JSON_DIR.glob(p):
            with open(fpath) as f:
                d = json.load(f)
            threshold = d["threshold"]
            compiler = BNCCompiler(d["network_name"], d["y"], threshold,
                                   d["features"])
            ret = compiler.compile_bnc()
            _logger.info(ret)
            summary_list.append(ret)
            _logger.info("---------------------------------")

# %%
if __name__ == "__main__":
    now_timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    _logger.info(f"Time now: {now_timestamp}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(funcName)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_DIR / f'{now_timestamp}.log')
        ])
    utils.cleanup_gv_files(config.OUT_DIR)
    main()
