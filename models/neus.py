import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch,validate_empty_rays
from systems.utils import update_module_step
from nerfacc import render_weight_from_density, render_weight_from_alpha, accumulate_along_rays, OccGridEstimator, ray_aabb_intersect

from pytorch_lightning.utilities.rank_zero import rank_zero_only

@rank_zero_only
def rank_zero_print(print_str):
    print(print_str)

class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = 'AABB'
        self.near_plane, self.far_plane = 0.0, 10.0*self.config.radius
        self.prune_alpha=self.config.get("prune_alpha",False)
        self.prune_alpha_threshold=self.config.get('prune_alpha_threshold',0.01)
        
        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = 'UN_BOUNDED_SPHERE'
            self.near_plane_bg, self.far_plane_bg = 0.0, 1e10
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01            

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        # if self.config.grid_prune:
        self.occupancy_grid = OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=128,
            levels=1
        )
        if not self.config.grid_prune:
            self.occupancy_grid.occs.fill_(True)
            self.occupancy_grid.binaries.fill_(True)
        if self.config.learned_background:
            self.occupancy_grid_bg = OccGridEstimator(
                roi_aabb=self.scene_aabb,
                resolution=256,
                levels=1
            )
            if not self.config.grid_prune:
                self.occupancy_grid_bg.occs.fill_(True)
                self.occupancy_grid_bg.binaries.fill_(True)
        self.randomized = self.config.randomized
        self.randomized_fg = self.config.get('randomized_fg',False)
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[...,None] * self.render_step_size_bg
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.update_every_n_steps(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
            if self.config.learned_background:
                self.occupancy_grid_bg.update_every_n_steps(step=global_step, occ_eval_fn=occ_eval_fn_bg, occ_thre=self.config.get('grid_prune_occ_thre_bg', 0.01))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_bg_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.
            density, _ = self.geometry_bg(positions)
            return density.view(-1)

        # if the ray intersects with the bounding box, start from the farther intersection point
        # otherwise start from self.far_plane_bg
        # note that in nerfacc t_max is set to 1e10 if there is no intersection
        # near_plane_bg = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        _, t_max, hits = ray_aabb_intersect(
            rays_o=rays_o,
            rays_d=rays_d,
            aabbs=self.scene_aabb.unsqueeze(0),
            near_plane=self.near_plane_bg,
            far_plane=self.far_plane_bg,
        )
        t_max, hits = t_max.squeeze(), hits.squeeze()
        t_max = torch.where(hits,t_max,0.0)
        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.occupancy_grid_bg.sampling(
                rays_o, rays_d,
                sigma_fn=sigma_fn,
                near_plane=self.near_plane_bg,
                far_plane=self.far_plane_bg,
                t_min=t_max,
                render_step_size=self.render_step_size_bg,
                stratified=self.randomized,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0
            )
        
        ray_indices = ray_indices.flatten().long()
        ray_indices, t_starts, t_ends=validate_empty_rays(ray_indices,t_starts,t_ends)
        t_starts = t_starts.flatten()[..., None]
        t_ends = t_ends.flatten()[..., None]
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        # mask=(midpoints.flatten()<t_min[ray_indices]) | (midpoints.flatten()>t_max[ray_indices])
        # rank_zero_print("t_origins.shape={}, t_dirs.shape={}, midpoints.shape={}, mask.shape={}, t_min.shape={}, t_max.shape={}".format(
        #     t_origins.shape,t_dirs.shape,midpoints.shape,mask.shape,t_min.shape,t_max.shape
        # ))
        # midpoints=midpoints[mask]
        positions = t_origins + t_dirs * midpoints
        intervals = t_ends - t_starts

        density, feature = self.geometry_bg(positions) 
        rgb = self.texture_bg(feature, t_dirs)

        weights, trans_, _  = render_weight_from_density(
            t_starts[..., 0], t_ends[..., 0], 
            density.view(-1), 
            ray_indices=ray_indices, 
            n_rays=n_rays)
        opacity = accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)
        depth = accumulate_along_rays(weights, values=midpoints, ray_indices=ray_indices, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, values=rgb, ray_indices=ray_indices, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)       

        out = {
            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })

        return out

    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        
        def alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.
            sdf = self.geometry.forward_level(positions)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
            return alpha.view(-1)

        t_min, t_max, hits = ray_aabb_intersect(
            rays_o=rays_o,
            rays_d=rays_d,
            aabbs=self.scene_aabb.unsqueeze(0),
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )
        t_min, t_max, hits = t_min.squeeze(), t_max.squeeze(), hits.squeeze()

        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
                rays_o, rays_d,
                alpha_fn=alpha_fn if self.prune_alpha else None,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                t_min=t_min,
                t_max=t_max,
                render_step_size=self.render_step_size,
                stratified=self.randomized_fg,
                cone_angle=0.0,
                alpha_thre=self.prune_alpha_threshold if self.prune_alpha else 0.0
            )
        
        ray_indices = ray_indices.flatten().long()
        ray_indices, t_starts, t_ends=validate_empty_rays(ray_indices,t_starts,t_ends)
        t_starts = t_starts.flatten()[..., None]
        t_ends = t_ends.flatten()[..., None]
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        if self.config.geometry.grad_type == 'finite_difference':
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
        else:
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)
        rgb = self.texture(feature, t_dirs, normal)

        weights, _ = render_weight_from_alpha(
            alpha,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        opacity = accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)
        depth = accumulate_along_rays(weights, values=midpoints, ray_indices=ray_indices, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, values=rgb, ray_indices=ray_indices, n_rays=n_rays)

        comp_normal = accumulate_along_rays(weights, values=normal, ray_indices=ray_indices, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        out = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1)                
            })
            if self.config.geometry.grad_type == 'finite_difference':
                out.update({
                    'sdf_laplace_samples': sdf_laplace
                })

        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
            mesh['v_rgb'] = rgb.cpu()
        return mesh
