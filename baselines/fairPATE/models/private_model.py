from ..architectures.celeba_net import CelebaNet
from ..architectures.celebasensitive_net import CelebaSensitiveNet
from ..architectures.resnet_pretrained import pretrained_resnet50
from .utils_models import get_model_type_by_id, get_model_name_by_id


def get_private_model(name, model_type, args):
    """Private model held by each party."""
    if model_type == 'resnet50_pretrained':
            model = pretrained_resnet50()
    elif model_type == 'CelebaNet':
        model = CelebaNet(name=name, args=args)
    elif model_type == 'CelebaSensitiveNet':
        model = CelebaSensitiveNet(name=name, args=args)
    else:
        raise Exception(f'Unknown architecture: {model_type}')

    # Set the attributes if not already set.
    if getattr(model, 'dataset', None) == None:
        model.dataset = args.dataset
    if getattr(model, 'model_type', None) == None:
        model.model_type = model_type

    return model


def get_private_model_by_id(args, id=0):
    model_type = get_model_type_by_id(args=args, id=id)
    name = get_model_name_by_id(id=id)
    model = get_private_model(name=name, args=args, model_type=model_type)
    return model
