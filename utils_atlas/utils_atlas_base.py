from PIL import ImageOps, Image
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.utils import make_grid


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: transpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2numpy(input_image):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: transpose
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy


def interpolate_keys(keys, resx_ratio, resy_ratio):
    interpolated = torch.nn.functional.interpolate(keys.permute(0, 3, 1, 2), size=(resy_ratio, resx_ratio),
                                                   mode='bilinear', align_corners=False)  # shape [batch, emb, h, w]
    return interpolated.permute(0, 2, 3, 1)  # shape [batch, h, w, emb]


def extract_upsampled_keys(input_image, dino_extractor, dino_keys_layer=11, facet="key", feat_res=0, preprocess_image=True):
    if preprocess_image:
        if isinstance(input_image, torch.Tensor):
            prep_img = dino_extractor.normalize_image(input_image)
        else:  # PIL image
            prep_img = dino_extractor.preprocess_pil_image(input_image)
    else:
        prep_img = input_image
    descs_a = dino_extractor.extract_descriptors(prep_img.to(dino_extractor.device), layer=dino_keys_layer, facet=facet, include_cls=False)
    descs_a = descs_a[:, 0].reshape(prep_img.shape[0], *dino_extractor.num_patches, -1)  # [batch, resy, resx, emb]

    if feat_res != 0:
        descs_a_upsampled = interpolate_keys(descs_a, feat_res, feat_res)  # [batch, resy, resx, emb]
        descs_a_upsampled = descs_a_upsampled.permute(0, 3, 1, 2)  # [batch, emb, resy, resx]
        return descs_a_upsampled
    else:
        descs_a = descs_a.permute(0, 3, 1, 2)  # [batch, emb, resy, resx]
        return descs_a


def extract_dino_saliency_map(saliency_from_path, dino_extractor, load_size, feat_res=0):
    saliency = Image.open(saliency_from_path).convert("RGB")
    if feat_res != 0:
        saliency = saliency.resize((feat_res, feat_res), resample=Image.LANCZOS)
    else:
        resize_to_y = int(dino_extractor.p / dino_extractor.stride[0] * (load_size[0] // dino_extractor.p - 1) + 1)
        resize_to_x = int(dino_extractor.p / dino_extractor.stride[1] * (load_size[1] // dino_extractor.p - 1) + 1)
        saliency = saliency.resize((resize_to_x, resize_to_y), resample=Image.LANCZOS)
    frame_map = transforms.ToTensor()(ImageOps.grayscale(saliency)).unsqueeze(0).to(dino_extractor.device)  # shape [1, 1, resy, resx]
    processed_map = frame_map[0]  # [1, resy, resx]
    return processed_map.detach()


def get_keys_loss_func():
    return l2_loss


def l2_loss(x, y):
    return (x - y) ** 2


def feat_mse_loss(x, y, dim=1):
    return l2_loss(x, y).mean(dim=dim)


def cosine_distance(x, y, dim=1):
    return 1 - F.cosine_similarity(x, y, dim=dim)


def norm(t, dim=1):
    return F.normalize(t, dim=dim, eps=1e-10, p=2)


# from https://github.com/wpeebles/gangealing/blob/ffa6387c7ffd3f7de76bdc693dc2272e274e9bfd/models/spatial_transformers/spatial_transformer.py#L618
def normalize(points, res, out_res):
    return points.div(out_res - 1).add(-0.5).mul(2).mul((res - 1) / res)


def apply_pca(desc, pca_n_components, trained_pca=None):
    """ desc shape [N, embedding_size] """
    if trained_pca is None:
        pca = PCA(n_components=pca_n_components).fit(desc)
    else:
        pca = trained_pca
    pca_descriptors = pca.transform(desc)
    pca_descriptors = transforms.ToTensor()(pca_descriptors)[0]
    return pca_descriptors


def calculate_fw_matrix(affine_params):
    """ shape of affine_params is [N, 4], and contains (rot (radians), scale, tx, ty) """

    def construct_matrix(N, cos_rot, sin_rot, shift_x, shift_y, scale=1.):
        matrix = [scale * cos_rot, -scale * sin_rot, shift_x,
                  scale * sin_rot, scale * cos_rot, shift_y]
        matrix = torch.cat(matrix, dim=1)  # (N, 6)
        matrix = matrix.reshape(N, 2, 3)  # (N, 2, 3)
        return matrix

    rot = affine_params[:, 0].unsqueeze(-1)  # rot is after tanh() * pi
    scale = affine_params[:, 1].unsqueeze(-1)  # scale is after exp
    shift_x = affine_params[:, 2].unsqueeze(-1)
    shift_y = affine_params[:, 3].unsqueeze(-1)
    cos_rot = torch.cos(rot)  # [N, 1]
    sin_rot = torch.sin(rot)
    backward_mat_noS = construct_matrix(affine_params.shape[0], cos_rot, sin_rot, shift_x, shift_y)
    R_mat = backward_mat_noS[:, :, :2]  # [N, 2, 2]
    txty = backward_mat_noS[:, :, 2][..., None]  # [N, 2, 1]
    forward_matrix = (
            (1 / scale.unsqueeze(-1)) * torch.cat((R_mat.permute(0, 2, 1), -R_mat.permute(0, 2, 1) @ txty), dim=-1))  # (N, 2, 3)
    return forward_matrix


def plot_images_grid(images_tensor, nrow=1, padding=3, pad_value=1, split=False, **grid_kwargs):
    if split:
        images_tensor_split = torch.split(images_tensor, images_tensor.shape[-2], dim=-1)
        images_collection_tensor = torch.cat(images_tensor_split, dim=0)
    else:
        images_collection_tensor = images_tensor
    image_grid = make_grid(images_collection_tensor, nrow=nrow, padding=padding, pad_value=pad_value, **grid_kwargs)
    return image_grid[None, ...]


def apply_griddata(input_image, upoints, edit_image_points_rgba, out_indices):
    propagated_edit = griddata(upoints, edit_image_points_rgba, (out_indices[:, 1], out_indices[:, 0]), method='linear')
    propagated_edit_t = transforms.ToTensor()(propagated_edit).reshape(1, input_image.shape[-2], input_image.shape[-1], 4).permute(0, 3, 1, 2)
    propagated_edit_t[propagated_edit_t.isnan()] = 0.
    a = propagated_edit_t[:, 3].unsqueeze(0)
    edited_image = input_image * (1 - a) + a * propagated_edit_t[:, :3]
    return edited_image


def upload_trained_model_and_data(checkpoint_path, atlas_model_module, dataset_module, device, eval_mode=True, upload_masks=True):
    checkpoint = torch.load(checkpoint_path)
    run_config = checkpoint["config"]
    if not upload_masks:
        run_config["use_masks"] = False

    dataset = dataset_module(run_config, device)
    if run_config["ext_horizontal_flips"]:
        dataset.images_state = checkpoint["images_state"]

    init_with_flow = eval_mode or (checkpoint["epoch"] >= run_config["bootstrap_stn_sim"])
    model = atlas_model_module(run_config, dataset.dino_emb_size, dataset.init_atlas_dict, device, init_with_flow=init_with_flow)
    if "atlas_params_requires_grad" in run_config and not run_config["atlas_params_requires_grad"]:  # relevant only for bicycle set of SPair-71K (fixed atlas)
        model.atlas_keys, model.atlas_saliency = dataset.imgs_dino_keys[run_config["im_idx_atlas_init"]].to(device), dataset.imgs_saliency_maps[run_config["im_idx_atlas_init"]].squeeze().to(device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    if eval_mode:
        model.eval()
    return checkpoint, run_config, dataset, model


def upload_checkpoint_for_training(config, atlas_model_module, dataset_module, device):
    checkpoint_path = config["checkpoint_path"]
    checkpoint, run_config, dataset, model = upload_trained_model_and_data(config["checkpoint_path"],
                                                                           atlas_model_module, dataset_module, device,
                                                                           eval_mode=False)
    model.train()
    upload_iter = checkpoint["epoch"]
    total_iter = config["n_iters"]
    bootstrapping_stn_sim = config["bootstrap_stn_sim"]
    save_model_starting_epoch = config["save_model_starting_epoch"]
    config.update(run_config)
    config["n_iters"] = total_iter - upload_iter
    config["bootstrap_stn_sim"] = 0 if upload_iter >= bootstrapping_stn_sim else (bootstrapping_stn_sim - upload_iter)
    print("bootstrap_stn_sim: ", config["bootstrap_stn_sim"])
    config["save_model_starting_epoch"] = 0 if upload_iter > save_model_starting_epoch else (save_model_starting_epoch - upload_iter)
    config["upload_trained_model_checkpoint"] = True
    config["checkpoint_path"] = checkpoint_path
    return checkpoint, config, dataset, model


def load_state_opt_criterion(config, checkpoint, criterion, optimizer_atlas, optimizer_stns):
    optimizer_stns.load_state_dict(checkpoint['stn_optimizer_state_dict'])
    optimizer_atlas.load_state_dict(checkpoint['atlas_optimizer_state_dict'])
    if config["ext_gradual_atlas_training"]:
        criterion.imgs_idx_to_update_atlas = checkpoint['imgs_idx_to_update_atlas']
        criterion.all_imgs_update_atlas = checkpoint['all_imgs_update_atlas']
    return criterion, optimizer_atlas, optimizer_stns
