import os
import torch


def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def _model_path_name(args):
    return f"{args.output_dir}/{args.dataset}/{args.model_name}_llm_frozen_{args.llm_frozen}_{args.num_epochs}_epochs_checkpoint_best.pth"


def save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)

    param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }
    path = _model_path_name(args)
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)


def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    checkpoint_path = _model_path_name(args)
    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model


def _reload_model(model, checkpoint_path):
    """
    Load the best checkpoint for evaluation.
    """

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model
