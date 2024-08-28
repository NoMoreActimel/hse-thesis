from core.featureshift_trainer import FeatureShiftDomainAdaptationTrainer

from omegaconf import OmegaConf

from core.utils.common import setup_seed
from core.utils.arguments import load_config
from pprint import pprint


def train(config):
    # pprint(OmegaConf.to_container(config))
    setup_seed(config.seed)
    trainer = FeatureShiftDomainAdaptationTrainer(config)
    trainer.train_loop()

if __name__ == "__main__":
    config = load_config()
    train(config)
