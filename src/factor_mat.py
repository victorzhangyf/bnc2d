import dataclasses, itertools, typing, functools
import numpy as np
import numpy.typing as npt

import global_config

config = global_config.Config()
np.set_printoptions(precision=5)


@dataclasses.dataclass
class FactorMat:
    # vs is always sorted
    mat: npt.NDArray[np.float32]
    vs: tuple[int, ...]

    # below all for debug
    nary: tuple[int, ...]
    var_names: tuple[str, ...]
    cond_vars: frozenset[int] = dataclasses.field(default_factory=frozenset)

    N_VARS: typing.ClassVar[int] = 0
    N_FEAS: typing.ClassVar[int] = 0
    CHECK_SUM_OUT_FEATURE: typing.ClassVar[bool] = False

    @staticmethod
    @functools.cache
    def vs_to_sorted_axis(vs: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(np.searchsorted(sorted(vs), vs).tolist())

    def _print_as_tabular(self) -> str:
        assert config.DEBUG_LVL > 0
        # iterate all combinations, based on vs order
        lines = [self._print_description()]
        lines.append("\t".join(self.var_names))
        indices = [range(i) for i in self.mat.shape]
        coords = itertools.product(*indices)
        for coord in coords:
            idx = tuple(indices[i][j] for i, j in enumerate(coord))
            line = []
            line.append(self.mat[idx])
            lines.append("\t\t".join(map(str, line)))
        return "\n".join(lines)

    def get_history(self) -> str | tuple:
        assert config.DEBUG_LVL > 0
        return self._print_description()

    def _print_description(self) -> str:
        assert config.DEBUG_LVL > 0
        if not self.vs:
            return "CONST"
        lv, rv = [], []
        for i, j in enumerate(self.vs):
            name = self.var_names[i]
            if j in self.cond_vars:
                rv.append(name)
            else:
                lv.append(name)
        left_var_str = ', '.join(lv)
        s = f"P ( {left_var_str}"
        if rv:
            right_var_str = ', '.join(rv)
            s = f"{s} | {right_var_str}"
        return s + " )"

    def _validate_shape(self):
        if not self.vs:
            assert self.mat.shape == (1, )
        for i, j in zip(self.mat.shape, self.nary):
            assert i <= j

    def __post_init__(self) -> None:
        self.mat.flags.writeable = False
        if config.DEBUG_LVL > 1:
            self._validate_shape()
            if not self.vs:
                assert self.mat.shape == (1, )
                return

            assert not np.any(np.isnan(self.mat))
            assert not np.any(np.isinf(self.mat))
            assert len(self.vs) > 0
            assert len(self.mat.shape) == len(self.vs)
            assert len(set(self.vs)) == len(self.vs)
            assert tuple(sorted(self.vs)) == self.vs
            assert self.cond_vars <= set(self.vs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FactorMat):
            raise NotImplementedError
        if self.vs != other.vs:
            return False
        return np.allclose(self.mat, other.mat)

    @staticmethod
    @functools.cache
    def _sum_out_axis(vs: tuple[int, ...], sum_out_indices: frozenset[int],
                      cond: frozenset[int], nary: tuple[int, ...],
                      names: tuple[str, ...]):
        summed_axis = tuple(
            np.searchsorted(vs, tuple(sum_out_indices)).tolist())
        remained_indices = tuple(sorted(set(vs) - sum_out_indices))

        new_nary, new_cond, new_names = (), frozenset(), ()
        if config.DEBUG_LVL > 1:
            remain = tuple(np.searchsorted(vs, remained_indices).tolist())
            new_cond = cond - sum_out_indices
            new_nary = tuple(nary[i] for i in remain)
            new_names = tuple(names[i] for i, j in enumerate(vs)
                              if j not in sum_out_indices)

        return summed_axis, remained_indices, new_cond, new_nary, new_names

    def sum_out(self, sum_out_indices: frozenset[int]) -> 'FactorMat':
        if not sum_out_indices:
            return self
        if not self.vs:
            raise NotImplementedError
        if config.DEBUG_LVL > 1:
            assert sum_out_indices <= set(self.vs)
            if FactorMat.CHECK_SUM_OUT_FEATURE:
                assert min(sum_out_indices) >= FactorMat.N_FEAS
        summed_axis_indices, remained_indices, cond_vars, nary, var_names = FactorMat._sum_out_axis(
            self.vs, sum_out_indices, self.cond_vars, self.nary,
            self.var_names)
        mat = np.atleast_1d(np.sum(self.mat, axis=summed_axis_indices))
        return FactorMat(mat=mat,
                         vs=remained_indices,
                         cond_vars=cond_vars,
                         nary=nary,
                         var_names=var_names)

    @staticmethod
    @functools.cache
    def _factor_mult_align_axis(
        vs1: tuple[int, ...], vs2: tuple[int, ...], cond1: frozenset[int],
        cond2: frozenset[int], nary1: tuple[int, ...], nary2: tuple[int, ...],
        names1: tuple[str, ...], names2: tuple[str, ...], divide: bool
    ) -> tuple[tuple[int, ...], list[int], list[int], frozenset[int], tuple[
            int, ...], tuple[str, ...]]:
        vset1 = frozenset(vs1)
        vset2 = frozenset(vs2)
        new_vset = vset1 | vset2
        new_vs = tuple(sorted(new_vset))
        new_axis1, new_axis2 = [], []
        if len(new_vset) > len(vset1):
            new_axis1 = np.searchsorted(new_vs,
                                        tuple(new_vset - vset1)).tolist()

        if len(new_vset) > len(vset2):
            new_axis2 = np.searchsorted(new_vs,
                                        tuple(new_vset - vset2)).tolist()
        nary, var_names, new_cond = (), (), frozenset()
        if config.DEBUG_LVL > 1:
            if divide:
                assert vset1 >= vset2
            left_vars1, left_vars2 = vset1 - cond1, vset2 - cond2
            if divide:
                new_cond = (cond1 - cond2) | left_vars2
            else:
                assert not (left_vars1 & left_vars2)
                new_cond = (cond1 - left_vars2) | (cond2 - left_vars1)
            nary1_d = {i: (j, k) for i, j, k in zip(vs1, nary1, names1)}
            nary2_d = {i: (j, k) for i, j, k in zip(vs2, nary2, names2)}
            nary, var_names = tuple(
                zip(*tuple(j for _, j in sorted((nary1_d | nary2_d).items()))))

        return new_vs, new_axis1, new_axis2, new_cond, nary, var_names

    def factor_mult(self,
                    other: 'FactorMat',
                    divide=False,
                    normalize=False) -> 'FactorMat':
        if not self.vs or not other.vs:
            if self.vs:
                vs = self.vs
                nary = self.nary
                var_names = self.var_names
                cond_vars = self.cond_vars
            else:
                vs = other.vs
                nary = other.nary
                var_names = other.var_names
                cond_vars = other.cond_vars
            return FactorMat(mat=np.atleast_1d(other.mat * self.mat),
                             vs=vs,
                             nary=nary,
                             var_names=var_names,
                             cond_vars=cond_vars)
        new_vs, new_axis1, new_axis2, new_cond, nary, var_names = FactorMat._factor_mult_align_axis(
            self.vs, other.vs, self.cond_vars, other.cond_vars, self.nary,
            other.nary, self.var_names, other.var_names, divide)
        mat1, mat2 = self.mat, other.mat
        if new_axis1:
            mat1 = np.expand_dims(mat1, axis=new_axis1)
        if new_axis2:
            mat2 = np.expand_dims(mat2, axis=new_axis2)
        if divide:
            mat = mat1 / mat2
            np.nan_to_num(mat, copy=False)
        else:
            mat = mat1 * mat2
        if normalize:
            mat = mat / np.sum(mat)
        return FactorMat(mat=mat,
                         vs=new_vs,
                         cond_vars=new_cond,
                         nary=nary,
                         var_names=var_names)

    def project_to(self, remained_var_set: frozenset[int]) -> 'FactorMat':
        sum_out_var_set = frozenset(self.vs) - remained_var_set
        return self.sum_out(sum_out_var_set)

    def condition_on(self, cond_var_set: frozenset[int]) -> 'FactorMat':
        if config.DEBUG_LVL > 0:
            assert cond_var_set
            assert cond_var_set < set(self.vs)
            assert not (cond_var_set & self.cond_vars)

        new_cond_vars = self.cond_vars | cond_var_set
        remained_var_set = frozenset(self.vs) - cond_var_set
        sum_out_axis = tuple(
            np.searchsorted(self.vs, tuple(remained_var_set)).tolist())
        summed = np.sum(self.mat, axis=sum_out_axis, keepdims=True)
        mat = np.divide(self.mat, summed)
        np.nan_to_num(mat, copy=False)
        assert mat.shape == self.mat.shape

        return FactorMat(mat=mat,
                         vs=self.vs,
                         cond_vars=new_cond_vars,
                         nary=self.nary,
                         var_names=self.var_names)


FactorMatTrue = FactorMat(mat=np.ones(1, dtype=np.float32),
                          vs=(),
                          nary=(),
                          var_names=(),
                          cond_vars=frozenset())

FactorMatFalse = FactorMat(mat=np.zeros(1, dtype=np.float32),
                           vs=(),
                           nary=(),
                           var_names=(),
                           cond_vars=frozenset())
