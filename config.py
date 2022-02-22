import os
import toml


class BaseConfig:
    def __init__(self, path_to_toml_config: str = None):
        if path_to_toml_config is None:
            self.config = self.automatically_get_config()
        else:
            self.config = toml.load(path_to_toml_config)

    @staticmethod
    def automatically_get_config():
        curdir = os.path.dirname(__file__)
        path_to_toml_config = os.path.join(curdir, "config.toml")
        return toml.load(path_to_toml_config)


class DataConfig(BaseConfig):
    def __init__(self, path_to_config: str = None):
        super().__init__(path_to_config)
        data_config = self.config["data"]
        self.data_url = data_config["data_url"]


class RandomForestConfig(BaseConfig):
    def __init__(self, path_to_config: str = None):
        super().__init__(path_to_config)
        models_config = self.config["models"]
        self.hyperparameters = models_config["random_forest"]["hyperparameters"]


class Config(BaseConfig):
    def __init__(self, path_to_config: str = None):
        super().__init__(path_to_config)
        self.data_config = DataConfig()
        self.random_forest_config = RandomForestConfig()
