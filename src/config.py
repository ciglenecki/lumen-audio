import hydra
from omegaconf import DictConfig, OmegaConf

hydra.initialize(version_base=None, config_path="../config")
_cfg = hydra.compose(config_name="config")
_cfg = OmegaConf.to_container(_cfg, resolve=True)
cfg = OmegaConf.create(_cfg)
print("Config:", OmegaConf.to_yaml(cfg))
