import yaml


class GlobalConfig:
    
    model_name :str= "Qwen/Qwen3-4B-AWQ" #"unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit"
    output_path: str = "output/"
    
    def __init__(
        self,
        config_path: str = "config/default.yaml"
    ):
        conf:dict= yaml.safe_load(open(config_path, "r"))
        if not conf:
            print("[WARNING] GlobalConfig has NO config abstracted correctly, \nusing default!")
            return
        print(conf)
        for (k,v) in conf.items():
            if hasattr(self,k): setattr(self,k,v)
