import torch
import numpy as np
import os
import glob
import random
import matplotlib
import imageio

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, to_rgb
from scipy.spatial import cKDTree  # <-- ADDED for performance

from analysis.molecule_builder import get_bond_order


##############
### Files ####
###########-->


def save_xyz_file(path, one_hot, positions, atom_decoder, id_from=0,
                  name='molecule', batch_mask=None):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if batch_mask is None:
        batch_mask = torch.zeros(len(one_hot))

    for batch_i in torch.unique(batch_mask):
        cur_batch_mask = (batch_mask == batch_i)
        n_atoms = int(torch.sum(cur_batch_mask).item())
        f = open(path + name + '_' + "%03d.xyz" % (batch_i + id_from), "w")
        f.write("%d\n\n" % n_atoms)
        atoms = torch.argmax(one_hot[cur_batch_mask], dim=1)
        batch_pos = positions[cur_batch_mask]
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = atom_decoder[atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, batch_pos[atom_i, 0], batch_pos[atom_i, 1], batch_pos[atom_i, 2]))
        f.close()


def load_molecule_xyz(file, dataset_info):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(dataset_info['atom_decoder']))
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, dataset_info['atom_encoder'][atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.xyz")
    if shuffle:
        random.shuffle(files)
    return files


# <----########
### Files ####
##############


# --- MODIFIED FUNCTION ---
def draw_sphere(ax, x, y, z, size, color, alpha):
    """
    Draws a 3D sphere with realistic lighting.
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Sphere coordinates
    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) # Removed * 0.8 hack for a true sphere
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))

    # --- ADDED: Create a light source ---
    light = LightSource(azdeg=120, altdeg=30)
    
    # Convert input color (hex) to an RGB tuple
    rgb_color = to_rgb(color)
    
    # Create an RGBA array for the surface facecolors
    # This allows the light source to shade the sphere
    colors_rgba = np.zeros(list(xs.shape) + [4])
    colors_rgba[..., 0] = rgb_color[0]
    colors_rgba[..., 1] = rgb_color[1]
    colors_rgba[..., 2] = rgb_color[2]
    colors_rgba[..., 3] = alpha

    # Plot the surface with shading
    ax.plot_surface(x + xs, y + ys, z + zs, 
                    rstride=2, cstride=2,
                    facecolors=colors_rgba,  # Use facecolors for shading
                    lightsource=light,       # Apply light source
                    shade=True,              # Enable shading
                    linewidth=0,
                    alpha=alpha)


# --- MODIFIED FUNCTION ---
def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color,
                  dataset_info, override_color=None):
    
    # --- SAFETY FIX ---
    # Clip the atom types to be within the valid range of our data arrays
    max_atom_index = len(dataset_info['radius_dic']) - 1
    safe_atom_type = np.clip(atom_type, 0, max_atom_index)
    
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 200 * radius_dic ** 2
    radii = radius_dic[safe_atom_type]
    areas = area_dic[safe_atom_type]

    # --- COLOR OVERRIDE LOGIC ---
    if override_color:
        colors = [override_color] * len(positions)
    else:
        colors_dic = np.array(dataset_info['colors_dic'])
        safe_color_atom_type = np.clip(atom_type, 0, len(colors_dic) - 1)
        colors = colors_dic[safe_color_atom_type]
    # --- END COLOR OVERRIDE LOGIC ---

    # Get positions as a numpy array for plotting and k-d tree
    all_positions_np = positions.cpu().numpy()
    x = all_positions_np[:, 0]
    y = all_positions_np[:, 1]
    z = all_positions_np[:, 2]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            # Use .item() for individual coordinates
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha,
                   c=colors)

    # --- PERFORMANCE IMPROVEMENT: Use k-d tree for bond finding ---
    # This is much faster than the old O(N^2) nested loop
    
    # 1. Create the tree from atom positions
    tree = cKDTree(all_positions_np)

    # 2. Set a max bond distance (Angstroms) to search for pairs
    # This is a heuristic; get_bond_order will do the precise check
    MAX_BOND_DIST = 3.5 

    # 3. Find all pairs of atoms within this distance
    bond_pairs = tree.query_pairs(r=MAX_BOND_DIST)

    # 4. Iterate *only* over these potential bond pairs
    for (i, j) in bond_pairs:
        p1 = all_positions_np[i]
        p2 = all_positions_np[j]
        dist = np.sqrt(np.sum((p1 - p2) ** 2))
        
        # Ensure atom types are valid before decoding
        atom1_idx = np.clip(atom_type[i], 0, len(dataset_info['atom_decoder']) - 1)
        atom2_idx = np.clip(atom_type[j], 0, len(dataset_info['atom_decoder']) - 1)
        
        atom1 = dataset_info['atom_decoder'][atom1_idx]
        atom2 = dataset_info['atom_decoder'][atom2_idx]
        
        draw_edge_int = get_bond_order(atom1, atom2, dist)
        line_width = 2

        draw_edge = draw_edge_int > 0
        if draw_edge:
            linewidth_factor = 1.5 if draw_edge_int == 4 else 1
            # Plot using the numpy positions
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    linewidth=line_width * linewidth_factor,
                    c=hex_bg_color, alpha=alpha)


# --- MODIFIED FUNCTION ---
def plot_molecule_and_pocket(
    positions_lig, atom_type_lig, positions_pocket, atom_type_pocket,
    dataset_info, camera_elev=0, camera_azim=0, save_path=None,
    spheres_3d=False, bg='black'):
    
    black = (0, 0, 0)
    white = (1, 1, 1)
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    
    if bg == 'black':
        ax.set_facecolor(black)
        ax.xaxis.line.set_color("black")
        ligand_bond_color = '#FFFFFF'
    else:
        ax.set_facecolor(white)
        ax.xaxis.line.set_color("white")
        ligand_bond_color = '#666666'

    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    # 1. Plot the pocket first with gray colors and lower alpha
    pocket_color = '#808080' # Gray
    plot_molecule(ax, positions_pocket, atom_type_pocket, alpha=0.4,
                  spheres_3d=spheres_3d, hex_bg_color=pocket_color,
                  dataset_info=dataset_info, override_color=pocket_color)

    # 2. Plot the ligand on top with its original colors
    plot_molecule(ax, positions_lig, atom_type_lig, alpha=1.0,
                  spheres_3d=spheres_3d, hex_bg_color=ligand_bond_color,
                  dataset_info=dataset_info, override_color=None)

    # Set axis limits based on the combined system
    all_positions = torch.cat([positions_lig, positions_pocket], dim=0)
    max_value = all_positions.abs().max().item()
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        
        # --- REMOVED ---
        # The np.clip(img * 1.4, ...) hack is no longer needed
        # because the LightSource in draw_sphere provides better lighting.
        
    else:
        plt.show()
    plt.close()


# --- MODIFIED FUNCTION ---
def plot_data3d(positions, atom_type, dataset_info, camera_elev=0,
                camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1.):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.xaxis.line.set_color("black")
    else:
        ax.xaxis.line.set_color("white")

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                  hex_bg_color, dataset_info)

    max_value = positions.abs().max().item()
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    
    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        # --- REMOVED ---
        # The np.clip(img * 1.4, ...) hack is no longer needed
        # because the LightSource in draw_sphere provides better lighting.
        
    else:
        plt.show()
    plt.close()


# --- MODIFIED FUNCTION ---
def plot_data3d_uncertainty(
        all_positions, all_atom_types, dataset_info, camera_elev=0,
        camera_azim=0,
        save_path=None, spheres_3d=False, bg='black', alpha=1.):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.xaxis.line.set_color("black")
    else:
        ax.xaxis.line.set_color("white")

    for i in range(len(all_positions)):
        positions = all_positions[i]
        atom_type = all_atom_types[i]
        plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                      hex_bg_color, dataset_info)

    if 'qm9' in dataset_info['name']:
        max_value = all_positions[0].abs().max().item()
        axis_lim = min(40, max(max_value + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'geom':
        max_value = all_positions[0].abs().max().item()
        axis_lim = min(40, max(max_value / 2 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'pdbbind':
        max_value = all_positions[0].abs().max().item()
        axis_lim = min(40, max(max_value / 2 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(dataset_info['name'])

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        # --- REMOVED ---
        # The np.clip(img * 1.4, ...) hack is no longer needed
        # because the LightSource in draw_sphere provides better lighting.
        
    else:
        plt.show()
    plt.close()


def plot_grid():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, [im1, im2, im3, im4]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()


def visualize(path, dataset_info, max_num=25, wandb=None, spheres_3d=False):
    files = load_xyz_files(path)[0:max_num]
    for file in files:
        positions, one_hot = load_molecule_xyz(file, dataset_info)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        dists = torch.cdist(positions.unsqueeze(0),
                            positions.unsqueeze(0)).squeeze(0)
        dists = dists[dists > 0]
        
        plot_data3d(positions, atom_type, dataset_info=dataset_info,
                    save_path=file[:-4] + '.png',
                    spheres_3d=spheres_3d)

        if wandb is not None:
            path = file[:-4] + '.png'
            im = plt.imread(path)
            wandb.log({'molecule': [wandb.Image(im, caption=path)]})


def visualize_chain(path, dataset_info, wandb=None, spheres_3d=False,
                    mode="chain"):
    files = load_xyz_files(path)
    files = sorted(files)
    save_paths = []

    for i in range(len(files)):
        file = files[i]

        positions, one_hot = load_molecule_xyz(file, dataset_info=dataset_info)

        atom_type = torch.argmax(one_hot, dim=1).numpy()
        fn = file[:-4] + '.png'
        plot_data3d(positions, atom_type, dataset_info=dataset_info,
                    save_path=fn, spheres_3d=spheres_3d, alpha=1.0)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images')
    
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


def visualize_chain_uncertainty(
        path, dataset_info, wandb=None, spheres_3d=False, mode="chain"):
    files = load_xyz_files(path)
    files = sorted(files)
    save_paths = []

    for i in range(len(files)):
        if i + 2 == len(files):
            break

        file = files[i]
        file2 = files[i + 1]
        file3 = files[i + 2]

        positions, one_hot, _ = load_molecule_xyz(file,
                                                  dataset_info=dataset_info)
        positions2, one_hot2, _ = load_molecule_xyz(
            file2, dataset_info=dataset_info)
        positions3, one_hot3, _ = load_molecule_xyz(
            file3, dataset_info=dataset_info)

        all_positions = torch.stack([positions, positions2, positions3], dim=0)
        one_hot = torch.stack([one_hot, one_hot2, one_hot3], dim=0)

        all_atom_type = torch.argmax(one_hot, dim=2).numpy()
        fn = file[:-4] + '.png'
        plot_data3d_uncertainty(
            all_positions, all_atom_type, dataset_info=dataset_info,
            save_path=fn, spheres_3d=spheres_3d, alpha=0.5)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images')
    
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


if __name__ == '__main__':
    # plot_grid()
    import qm9.dataset as dataset
    from configs.datasets_config import qm9_with_h, geom_with_h

    # Use 'macosx' for local testing, but 'Agg' is set at the top
    # for server-side compatibility.
    # matplotlib.use('macosx') 

    task = "visualize_molecules"
    task_dataset = 'geom'

    if task_dataset == 'qm9':
        dataset_info = qm9_with_h

        class Args:
            batch_size = 1
            num_workers = 0
            filter_n_atoms = None
            datadir = 'qm9/temp'
            dataset = 'qm9'
            remove_h = False

        cfg = Args()
        dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

        for i, data in enumerate(dataloaders['train']):
            positions = data['positions'].view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = data['one_hot'].view(-1, 5).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            plot_data3d(
                positions_centered, atom_type, dataset_info=dataset_info,
                spheres_3d=True)

    elif task_dataset == 'geom':
        files = load_xyz_files('outputs/data')
        # matplotlib.use('macosx')
        for file in files:
            x, one_hot, _ = load_molecule_xyz(file, dataset_info=geom_with_h)

            positions = x.view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = one_hot.view(-1, 16).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            mask = (x == 0).sum(1) != 3
            positions_centered = positions_centered[mask]
            atom_type = atom_type[mask]

            plot_data3d(
                positions_centered, atom_type, dataset_info=geom_with_h,
                spheres_3d=False)

    else:
        raise ValueError(dataset)