import numpy as np
import os
import time
import torch
import sys

from network import DMM_Decoder
from data.naca_2d import naca
from data.uiuc_2d import uiuc
from mesh_io.turbulent_plate_mesh import Turbulent_plate_mesh, read_points, read_contour_id_list
from loss_utils import batched_chamfer_distance, reg_loss
from visualize_utils import plot_airfoil_2d_details

def do_minimization_iter(
            decoder,
            v,
            unique_surface_id_list,
            boundary_id_list,
            target_pts,
            optimizer,
            lr_scheduler,
            regular_sampling_ratio
           ):
    batch_size = target_pts.shape[0] if target_pts is not None else 1
    decoder.train()
    
    v_contour = v[unique_surface_id_list].repeat(batch_size, 1, 1)
    v_boundary = v[boundary_id_list].repeat(batch_size, 1, 1)


    num_sampled = int(regular_sampling_ratio * v.shape[0])
    rand_id = torch.randint(low=0, high=v.shape[0], device=v.device, size=[num_sampled])
    combined_id = torch.cat([unique_surface_id_list, rand_id])
    uniques, counts = combined_id.unique(return_counts=True)
    rand_id = uniques[counts == 1]
    v_sampled = v[rand_id].repeat(batch_size, 1, 1)
    v_sampled.requires_grad = True
    v_ = torch.cat([v_contour, v_sampled, v_boundary], dim=1)

    delta = decoder(v_.float())
    delta_contour = delta[:, :v_contour.shape[1], :]
    delta_sampled = delta[:, v_contour.shape[1]:v_contour.shape[1] + v_sampled.shape[1], :]
    delta_boundary = delta[:, v_contour.shape[1] + v_sampled.shape[1]:, :]
    # divide into lower and upper bound
    delta_boundary_1 = delta_boundary[:,:137,:]
    delta_boundary_2 = delta_boundary[:,137:,:]
    

    v_contour_deformed = v_contour + delta_contour

    # for the shape distance to the target
    # our target is all zero, remain unchanged
    # loss_contour = ((delta_contour ** 2).sum(dim=-1) ** 0.5).mean()
    # large weight to keep it unchanged
    #loss_chamfer = batched_chamfer_distance(
    #    v_contour_deformed, target_pts,
    #    single_sided_argmin_on_pt2=True,
    #    single_sided_argmin_on_pt1=True
    #)
    # the deforamtion quality loss
    loss_chamfer = ((torch.abs(v_contour_deformed - target_pts)).sum(dim=-1)).mean()
    loss_reg = reg_loss(v_sampled, delta_sampled)
    
    #loss_chamfer = torch.log10(loss_chamfer)
    

    #loss_chamfer = torch.clip(loss_chamfer + 16, 0)

    
    # the outer boundary change loss
    loss_boundary = ((delta_boundary ** 2).sum(dim=-1) ** 0.5).mean()

    #loss_boundary_1 = ((delta_boundary_1 ** 2).sum(dim=-1) ** 0.5).mean()
    loss_boundary_1 = ((torch.abs(delta_boundary_1)).sum(dim=-1)).mean()
    #loss_boundary_1 = torch.log10(loss_boundary_1) + 32
    #loss_boundary_1 = torch.clip(loss_boundary_1, 0)

    #loss_boundary_2 = ((delta_boundary_2 ** 2).sum(dim=-1) ** 0.5).mean()
    loss_boundary_2 = ((torch.abs(delta_boundary_2)).sum(dim=-1)).mean()
    #loss_boundary_2 = torch.log(loss_boundary_2) + 16
    #loss_boundary_2 = torch.clip(loss_boundary_2, 0)

    # todo: the first layer to the target position corresponding to y+

    #loss = 100 * loss_chamfer + loss_reg + 1 * loss_boundary_1 + 0.05 *loss_boundary_2

    loss = 1 * loss_boundary_1 + 1 * loss_chamfer + loss_boundary_2 + loss_reg

    loss.backward()
    optimizer.step()
    #lr_scheduler.step()
    

    return delta_contour, loss_chamfer, loss_reg, loss_boundary_1, loss_boundary_2, loss

#
# main
#
if __name__ == '__main__':
    #
    # init grid
    #
    CFD_mesh = Turbulent_plate_mesh(
        cache_name = './mesh_io/Scale_50_same_upper/init_mesh_cache.pymesh',
        su2Mesh_dir = './mesh_io/Scale_50_same_upper'
    )
    _ = CFD_mesh.init_mesh(use_cache = False)
    # load the meshs
    # v is the meshes, todo: get first layer list (implement like meshi_io/naca0012)
    v_old, _, edge_set, edge_index, faces, num_vertices, contour_id_list, boundary_id_list = \
                                            CFD_mesh.parse_mesh_dict(to_tensor=True)
    v = v_old
    v[:,1] *= 1e4
    print(v)
    
    unique_contour_id_list = contour_id_list.flatten().unique() + 137
    
    #
    # init model
    #
    decoder = DMM_Decoder(
        v_dim=2,
        layer_dim=[256, 512, 256]
    )
    print('Model info')
    print(decoder)
    decoder.init_weights()

    #
    # init cuda
    #
    use_cuda = False
    if use_cuda:
        unique_contour_id_list = unique_contour_id_list.cuda()
        boundary_id_list = boundary_id_list.cuda()
        decoder = decoder.cuda()
        v = v.cuda()

    # main
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', type=str, required=True, default='naca')
    parser.add_argument('-profile', type=str, default='3413')
    parser.add_argument('-reconIter', type=int, default='600')
    parser.add_argument('-workspaceDir', type=str, default='exp_dmm_2d/')
    parser.add_argument('-dumpModel', type=str, default='True')
    args = parser.parse_args()
    workspace_dir = args.workspaceDir
    dump_model = args.dumpModel
    assert(dump_model in ['True', 'False'])
    dump_model = True if dump_model == 'True'else False

    mesh_path= ""

    points = read_points("mesh_flatplate_turb_137x97.su2")
    contour_list =  read_contour_id_list("mesh_flatplate_turb_137x97.su2")

    # calculate new y+
    airfoil_pts = v[unique_contour_id_list,]
    airfoil_pts[:,1] = airfoil_pts[:,1] / 30

    airfoil_pts = airfoil_pts[None,:]
    
    
    

    num_recon_iters = args.reconIter
    optimizer = torch.optim.Adam(params = decoder.parameters(), lr = 3e-6)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500,1000], gamma=0.1)

    print('Reconstruct ...')
    tic = time.time()
    for i in range(num_recon_iters):
        optimizer.zero_grad()
        delta_contour, loss_chamfer, loss_reg, loss_boundary_1, loss_boundary_2, loss = do_minimization_iter(
            decoder,
            v,
            unique_contour_id_list,
            boundary_id_list,
            airfoil_pts,
            optimizer,
            lr_scheduler,
            regular_sampling_ratio = 0.02
        )
        if i % 10 == 0:
            print(' iter {}, loss: {:4e}, y+ distance: {:.4e}, reg loss: {:.4e}, boundary loss 1: {:.4e}, boundary loss 2: {:.4e}'.format(i, loss.item(), loss_chamfer.item(), loss_reg.item(), loss_boundary_1.item(), loss_boundary_2.item()))
    toc = time.time()
    print('Done in {}s'.format(toc - tic))

    decoder.eval()
    v_ = v.repeat(airfoil_pts.shape[0], 1, 1)
    v_ += decoder(v_.float())
    v_ = v_.detach().numpy()
    
    v_ = v_[0,:,:]
    print(v_[0,])
    print(v_[137,])
    print(v_[-1,])
    
    v_[:137,] = v_old[:137,].detach().numpy()
    #print(v_.shape)
    for i, j in enumerate(np.arange(0, v_.shape[0])):
        v_[j,0] = v_[i % 137, 0]
    #print(v_)

    CFD_mesh.write_to_su2Mesh(v_, "output_scale_50_same_upper.su2")
