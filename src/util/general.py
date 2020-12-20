from src.recommender.models.AMF import AMF
from src.recommender.models.BPRMF import BPRMF
from src.recommender.models.Random import Random


def format_error(error):
    ret = ""
    for line in error:
        ret += line
    return ret


def get_model(args, data, path_output_rec_result, path_output_rec_weight, path_output_rec_list):
    if args.rec == 'bprmf':
        return BPRMF(data, path_output_rec_result, path_output_rec_weight, path_output_rec_list, args)
    elif args.rec == 'amf':
        return AMF(data, path_output_rec_result, path_output_rec_weight, path_output_rec_list, args)
    elif args.rec == 'random':
        return Random(data, path_output_rec_result, path_output_rec_weight, path_output_rec_list, args)
    else:
        raise NotImplementedError('Unknown Recommender Model.')
