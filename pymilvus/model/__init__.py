import sys


class MilvusModelLoader:
    _milvus_model_loaded = False
    _milvus_model = None

    def __getattr__(self, name: str):
        if not self._milvus_model_loaded:
            self._load_milvus_model()
        try:
            return getattr(self._milvus_model, name)
        except AttributeError as e:
            err_str = (
                f"The attribute '{name}' is not found in 'pymilvus.model'. "
                "This might be due to an outdated version of 'milvus_model'. "
                "For upgrading to the latest version, use 'pip install milvus-model --upgrade'. "
                "For more information, please visit https://github.com/milvus-io/milvus-model."
            )
            raise AttributeError(err_str) from e

    def _load_milvus_model(self):
        try:
            import milvus_model

            self._milvus_model = milvus_model
        except ImportError as e:
            err_str = (
                "The 'milvus_model' package is not installed. "
                "For installation, use 'pip install pymilvus[model]'. "
                "For more information, please visit https://github.com/milvus-io/milvus-model."
            )
            raise ImportError(err_str) from e
        self._milvus_model_loaded = True


sys.modules[__name__] = MilvusModelLoader()
