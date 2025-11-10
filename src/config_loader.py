import yaml


class GlobalConfig:
    def __init__(
        self,
        config_path: str = "config/default.yaml"
    ):
        self.model_name :str= "Qwen/Qwen3-4B-AWQ"
        conf :dict= yaml.safe_load(open(config_path, "r"))
        if not conf:
            print("[WARNING] GlobalConfig has NO config abstracted correctly, \nusing default!")
            return
        for k,v in conf:
            if hasattr(self,k): setattr(self,k,v)
