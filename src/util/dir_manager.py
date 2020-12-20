import os
import shutil


def manage_directories(path_output_rec_result, path_output_rec_weight, path_output_rec_list):
    if os.path.exists(os.path.dirname(path_output_rec_result)):
        shutil.rmtree(os.path.dirname(path_output_rec_result))
    os.makedirs(os.path.dirname(path_output_rec_result))
    if os.path.exists(os.path.dirname(path_output_rec_weight)):
        shutil.rmtree(os.path.dirname(path_output_rec_weight))
    os.makedirs(os.path.dirname(path_output_rec_weight))
    if os.path.exists(os.path.dirname(path_output_rec_list)):
        shutil.rmtree(os.path.dirname(path_output_rec_list))
    os.makedirs(os.path.dirname(path_output_rec_list))

def get_paths(args, path_output_rec_result, path_output_rec_weight, path_output_rec_list):
    if args.rec == 'bprmf':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')

        path_output_rec_list = path_output_rec_list.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'XX',
                                                               'XX')

    elif args.rec == 'amf':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.adv_eps),
                                                               '' + args.adv_type)

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.adv_eps),
                                                               '' + args.adv_type)

        path_output_rec_list = path_output_rec_list.format(args.dataset,
                                                               args.rec,
                                                               'emb' + str(args.embed_size),
                                                               'ep' + str(args.epochs),
                                                               'eps' + str(args.adv_eps),
                                                               '' + args.adv_type)

    elif args.rec == 'random':
        path_output_rec_result = path_output_rec_result.format(args.dataset,
                                                               args.rec,
                                                               'XX',
                                                               'XX',
                                                               'XX',
                                                               'XX')

        path_output_rec_weight = path_output_rec_weight.format(args.dataset,
                                                               args.rec,
                                                               'XX',
                                                               'XX',
                                                               'XX',
                                                               'XX')

        path_output_rec_list = path_output_rec_list.format(args.dataset,
                                                               args.rec,
                                                               'XX',
                                                               'XX',
                                                               'XX',
                                                               'XX')

    return path_output_rec_result, path_output_rec_weight, path_output_rec_list
