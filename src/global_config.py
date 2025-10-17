import pathlib, typing


class Config:
    _shared_state: dict[str, typing.Any] = {"_initialized": False}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not self._initialized:
            self._init_consts()
            _default_debug_lvl = 0
            self.set_debug_lvl(_default_debug_lvl)
            self._initialized = True

    def _init_consts(self) -> None:
        self.ABSORB_NEIGHBOR_CLUSTER = False  
        self.ABSORB_NEIGHBOR_FEATURE = True
        self.MAX_LEAF_FEATURES_RATIO = lambda x: 1
        self.C2D_DT_BAL_RANGE = (10, 40)
        self.DISABLE_SEARCH_TREE = True
        self.DISABLE_DECIDE_NODE = False
        self.JOINT_TREE_QUALITY_CONSIDER_SEP_SIZE = True
        self.JOINT_TREE_QUALITY_CONSIDER_LEFT_LEAF_SIZE = True

        self.AC_DEVICE = 'cuda'
        self.AC_BATCH_SIZE = 1024
        self.FAC_SZ_LIMIT = 31

        self.LOG_DIR = pathlib.Path("log")

        self.NETWORK_BASE_DIR = pathlib.Path("networks")
        self.CACHE_DIR = pathlib.Path("tmp")
        self.OUT_DIR = pathlib.Path("out")

        self.PROBLEMS_JSON_DIR = pathlib.Path("problems")
        self.TARGET_CLS = 1
        self.DEBUG_CHECK_FEATURES_LIMIT = 22
        self.SIMPLIFY_NNF = False
        self.JOINT_TREE_METHOD = [
            # "c2d",
            "samiam",  
        ]
        self.C2D_BINARY = "./ace_v3.0_linux86/c2d_linux"

    def set_debug_lvl(self, l: int) -> None:
        assert l >= 0 and l <= 2
        self.DEBUG_LVL = l  

        self.DEBUG_INTERMEDIATE_PVZ_RESULT = False
        self.DEBUG_FAC_CHECK_ZERO = True
        self.DEBUG_CHECK_NNF = True
        self.DEBUG_PROFILE_CODE = False
        self.DEBUG_CHECK_ROOT_PROB = False
        self.DEBUG_READ_EXISTING_JTREE = False

        self.DEBUG_CHECK_DETERMINISTIC = not self.SIMPLIFY_NNF
        self.DEBUG_CHECK_DECOMPOSABLE = True
        if self.DEBUG_LVL < 2:
            self.DEBUG_INTERMEDIATE_PVZ_RESULT = False
            self.DEBUG_CHECK_ROOT_PROB = False
            self.DEBUG_READ_EXISTING_JTREE = False
            self.DEBUG_CHECK_DETERMINISTIC = False
            self.DEBUG_CHECK_DECOMPOSABLE = False
            self.DEBUG_FAC_CHECK_ZERO = False
            if self.DEBUG_LVL < 1:
                self.DEBUG_PROFILE_CODE = False
                self.DEBUG_CHECK_NNF = False

    def __hash__(self):
        return 1

    def __eq__(self, other):
        try:
            return self.__dict__ is other.__dict__
        except Exception:
            return 0
