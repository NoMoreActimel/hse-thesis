from core.evaluation import EvaluationManager
from core.utils.arguments import load_config
from core.utils.reading_weights import read_weights
from core.utils.example_utils import Inferencer
from examples.draw_util import weights

def evaluate(config):
    eval_manager = EvaluationManager(config)
    
    gan_domain = 'ffhq'
    s_domain = 'pop_art_indomain'

    ckpt = read_weights(weights[s_domain])
    ckpt_ffhq = {'sg2_params': ckpt['sg2_params']}
    ckpt_ffhq['sg2_params']['checkpoint_path'] = weights[gan_domain]

    model = 
    eval_manager.get_metrics()



if __name__ == "__main__":
    config = load_config()
    evaluate(config)
