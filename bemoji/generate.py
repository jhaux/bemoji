from PIL import Image, ImageDraw, ImageFont
import numpy as np
from time import sleep
import os

import multiprocessing as mp

from tqdm.autonotebook import tqdm, trange
import matplotlib.pyplot as plt


def dir2vec(direction):
    return np.array([np.sin(direction), np.cos(direction)])


def norm_pdf(X):
    return np.exp(-0.5*X**2) / np.sqrt(2*np.pi)


def prune(arr, amount=60):
    return arr[amount:-amount]


def linear_walk(total_length=300, imsize=[256, 256], speed=5, uniq_speed=1, off_speed=1, off_freq=10, seed=42, fs=29):
    if seed is not None:
        prng = np.random.RandomState(seed)
    else:
        prng = np.random
    direction = prng.uniform(-np.pi, np.pi)
    start = prng.uniform([0, 0], imsize)
    
    phase_shift = 0
    
    steps = [start]
    lat_steps = [start]
    lon_steps = [start]
    dirs = [direction]
    is_contact = [False]
    for i in range(total_length + 2 * 60):
        next_step = steps[-1] + speed * dir2vec(direction)
        
        at_border = False
        if next_step[1] < 0:
            direction += (np.pi - 2 * direction)
            next_step = steps[-1] + speed * dir2vec(direction)
            at_border = True
        elif next_step[0] > (imsize[0] - 1):
            direction -= 2*(direction)
            next_step = steps[-1] + speed * dir2vec(direction)
            at_border = True
        elif next_step[1] > (imsize[1] - 1):
            direction += (np.pi - 2*direction)
            next_step = steps[-1] + speed * dir2vec(direction)
            at_border = True
        elif next_step[0] < 0:
            direction -= 2*(direction)
            next_step = steps[-1] + speed * dir2vec(direction)
            at_border = True
        
        if at_border:
            phase_shift += off_freq / 2.
            is_contact += [True]
        else:
            is_contact += [False]
            
        lat_steps += [next_step + dir2vec(direction + np.pi / 2.) * off_speed * np.sin(i / off_freq + phase_shift)]
        lon_steps += [next_step + dir2vec(direction) * uniq_speed * np.sin(i / off_freq + phase_shift)]

        steps += [next_step]
        dirs += [direction]
        
    steps = np.array(steps)
    lat_steps = np.array(lat_steps)
    lon_steps = np.array(lon_steps)
    
    is_contact = np.array(is_contact)
    
    filter_kernel = norm_pdf(np.linspace(-2, 2, num=fs))
    filter_kernel = filter_kernel / filter_kernel.sum()
    
    smoothing = np.convolve(is_contact.astype(float), filter_kernel, mode='same')[:, None]
    smoothing *= fs / 2.
    
    smooth_steps = np.stack([np.convolve(steps[:, i], filter_kernel, mode='same') for i in range(2)], axis=-1)
    sstep = smoothing * smooth_steps + (1 - smoothing) * steps
    
    smooth_lat = np.stack([np.convolve(lat_steps[:, i], filter_kernel, mode='same') for i in range(2)], axis=-1)
    slat = smoothing * smooth_lat + (1 - smoothing) * lat_steps
    
    smooth_lon = np.stack([np.convolve(lon_steps[:, i], filter_kernel, mode='same') for i in range(2)], axis=-1)
    slon = smoothing * smooth_lon + (1 - smoothing) * lon_steps
    
    return {'raw': [prune(steps, 2*fs), prune(lat_steps, 2*fs), prune(lon_steps, 2*fs)],
            'smoothing': prune(smoothing, 2*fs),
            'smooth': [prune(smooth_steps, 2*fs), prune(smooth_lat, 2*fs), prune(smooth_lon, 2*fs)]}


def walk2im(points, text=None, image=None, size=20, color=(255, 0, 0), imsize=[256, 256]):
    images = []
    #font = ImageFont.truetype("/export/home/jhaux/.fonts/noto/NotoColorEmoji.ttf", fs)
    if text is not None:
        font = ImageFont.truetype("/export/home/jhaux/.fonts/OpenSans/OpenSansEmoji.ttf", size, encoding='unic')
    elif image is not None:
        if isinstance(image, str):
            image = Image.open(image)
        image = image.resize((size, size))

    for p in points:
        im = Image.new('RGB', imsize, (255, 255, 255))
        
        x, y = p - size / 2.
        if text is not None:
            d = ImageDraw.Draw(im)
            d.text((x, y), text, font=font, fill=color)
            del d
        elif image is not None:
            im.paste(image, (int(x), int(y)), mask=image)
        
        d = ImageDraw.Draw(im)
        sx, sy = np.array(imsize) - 1
        d.line([ 0,  0, sx,  0], fill=(0, 0, 0))
        d.line([ 0,  0,  0, sy], fill=(0, 0, 0))
        d.line([sx,  0, sx, sy], fill=(0, 0, 0))
        d.line([ 0, sy, sx, sy], fill=(0, 0, 0))
        del d
        
        # images += [im.resize((256, 256))]
        images += [im]
    return images


def save_fn(args):
    images, indices, saveroot = args
    for idx in indices:
        im = images[idx]
        savepath = os.path.join(saveroot, f'{idx:0>3d}.png')
        im.save(savepath)
        
def walk2im_par(args):
    points, size, imsize, image = args
    return walk2im(points, size=size, imsize=imsize, image=image)


def plot_trajs(base_traj, sick_traj, uniq_traj, finl_traj, saveroot, tseed, s):
    f, AX = plt.subplots(1, 4, figsize=(15, 5))

    [ax1, ax2, ax3, ax4] = AX

    ax1.scatter(base_traj[:, 0], base_traj[:, 1], marker='.', c=np.linspace(0, 1, num=len(base_traj)), cmap='viridis')
    ax2.scatter(sick_traj[:, 0], sick_traj[:, 1], marker='.', c=np.linspace(0, 1, num=len(base_traj)), cmap='viridis')
    ax3.scatter(uniq_traj[:, 0], uniq_traj[:, 1], marker='.', c=np.linspace(0, 1, num=len(base_traj)), cmap='viridis')
    ax4.scatter(finl_traj[:, 0], finl_traj[:, 1], marker='.', c=np.linspace(0, 1, num=len(base_traj)), cmap='viridis')

    for ax in AX.flatten():
        ax.set_aspect(1)
        ax.set_xlim(0, s)
        ax.set_ylim(0, s)

    savepath = os.path.join(saveroot, f'traj_{tseed}.png')

    f.savefig(savepath)
    f.clf()
        

def make_dataset(dset_root='/export/scratch/jhaux/Data/BEmoji/Data/',
                 image_root='/export/scratch/jhaux/Data/BEmoji/Emoji Choice',
                 only_labels=False):
    s = 1024
    imsize = [s, s]
    seq_length = 300
    sick_speed = 0.2 * s
    fs = 7

    nw = 100
    
    em_dir = image_root
    emojis = os.listdir(em_dir)
    emojis = sorted([os.path.join(em_dir, f) for f in emojis if f.endswith('.png') and 'Oreo' in f])

    emojis = emojis[:10]
    
    saveroot = dset_root
    os.makedirs(saveroot, exist_ok=True)
    
    health_levels = np.linspace(0, 1, num=5)
    
    sizes = s * np.linspace(0.1, 0.2, num=4)
    sizes = sizes.astype(int)
    
    base_speeds = 0.015 * s * sizes**(-2.) / sizes[0]**(-2.)
    
    prng = np.random.RandomState(42)
    
    uniq_speeds = s * prng.uniform(0.01, 0.25, size=len(emojis))
    uniq_freqs = prng.uniform(2, 20, size=len(emojis))
    
    trajectory_seeds = np.sort(np.concatenate([[0, 42, 1908], prng.uniform(43, 3312, size=len(emojis) - 3).astype(int)]))

    # emojis = prng.choice(emojis, size=30)

    np.save(os.path.join(saveroot, 'identities.npy'), np.arange(len(emojis)))
    np.save(os.path.join(saveroot, 'trajectory_seeds.npy'), trajectory_seeds)
    np.save(os.path.join(saveroot, 'sizes.npy'), sizes)
    np.save(os.path.join(saveroot, 'health_levels.npy'), health_levels)

    Q = mp.Queue()

    args = []
    for i in range(len(emojis)):
        # for j in range(len(sizes)):
            args += [[
                [emojis[i]],
                [uniq_speeds[i]],
                [uniq_freqs[i]],
                # [sizes[j]],
                # [base_speeds[j]],
                sizes,
                base_speeds,
                trajectory_seeds,
                health_levels,
                saveroot,
                seq_length,
                sick_speed,
                fs,
                imsize,
                s,
                [i],
                Q,
                only_labels]]

    print(len(args))

    ps = []
    for i in range(len(args)):
        ps += [mp.Process(target=_generate_in_parallel, args=[args[i]])]

    try:
        for p in ps:
            p.start()

        total = len(emojis) * len(sizes) * len(trajectory_seeds) * len(health_levels)
        pbar = tqdm(total=total, desc='All')
        i = 1
        while i < total:
            Q.get()
            pbar.update(1)
            i += 1

    except Exception as e:
        for p in ps:
            p.terminate()
        raise e
    finally:
        for p in ps:
            p.join()


def _generate_in_parallel(args):
    
    emojis, uniq_speeds, uniq_freqs, sizes, base_speeds, trajectory_seeds, health_levels, saveroot, seq_length, sick_speed, fs, imsize, s, identity, Q, only_labels = args
    ebar = emojis
    for _, [em, us, uf, i] in enumerate(zip(ebar, uniq_speeds, uniq_freqs, identity)):
        # ebar.set_description('E')
        visited = []

        sbar = sizes
        for size, speed in zip(sbar, base_speeds):
            # sbar.set_description(f'S: {size:0>3d}')

            tbar = trajectory_seeds
            for tseed in tbar:
                # tbar.set_description(f'T: {tseed:0>4d}')

                hbar = health_levels
                for h in hbar:
                    # hbar.set_description(f'H: {h:0.3f}')

                    name = f'id:{i:0>3d}/s:{size:0>3d}/t:{tseed:0>4d}/h:{h:0.3f}'
                    savepath = os.path.join(saveroot, name)
                    outfile = os.path.join(savepath, '000_vid.mp4')

                    if os.path.exists(outfile) and not only_labels:
                        Q.put(1)
                        continue

                    results = linear_walk(seq_length,
                                          speed=speed,
                                          off_speed=sick_speed,
                                          uniq_speed=us,
                                          off_freq=uf,
                                          seed=tseed,
                                          fs=fs,
                                          imsize=imsize)
                    base_traj = results['smooth'][0]
                    sick_traj = results['smooth'][1]
                    uniq_traj = results['smooth'][2]
                    
                    finl_traj = 0.5 * (h * base_traj + (1-h) * sick_traj) + 0.5 * uniq_traj
                    
                    # trajs = np.array_split(finl_traj, 100)
                    # ims_ = P.map(walk2im_par, [(trajs[i], size, imsize, em) for i in range(len(trajs))])
                    # ims = []
                    # for im in ims_:
                    #     ims += im
                    path_path = os.path.join(savepath, 'image_path.npy')
                    image_paths = savepath + '/' + np.char.array(
                        np.char.zfill(
                            np.arange(len(finl_traj)).astype(str), 
                            3
                            )
                        ) + '.png'

                    np.save(
                            path_path,
                            image_paths
                            )

                    arr_save_path = os.path.join(savepath, 'base_trajectory.npy')
                    np.save(
                            arr_save_path,
                            base_traj / s
                            )
                    arr_save_path = os.path.join(savepath, 'sick_trajectory.npy')
                    np.save(
                            arr_save_path,
                            sick_traj / s
                            )
                    arr_save_path = os.path.join(savepath, 'uniq_trajectory.npy')
                    np.save(
                            arr_save_path,
                            uniq_traj / s
                            )
                    arr_save_path = os.path.join(savepath, 'finl_trajectory.npy')
                    np.save(
                            arr_save_path,
                            finl_traj / s
                            )

                    arr_save_path = os.path.join(savepath, 'identity.npy')
                    np.save(
                            arr_save_path,
                            np.array([i] * len(finl_traj)).astype(int)
                            )

                    speed_arr = np.array([speed / s] * len(finl_traj))
                    np.save(
                            os.path.join(savepath, 'speed.npy'),
                            speed_arr
                            )

                    uspeed_arr = np.array([us / s] * len(finl_traj))
                    np.save(
                            os.path.join(savepath, 'unique_speed.npy'),
                            uspeed_arr
                            )
                    freq_arr = np.array([uf] * len(finl_traj))
                    np.save(
                            os.path.join(savepath, 'unique_frequencey.npy'),
                            freq_arr
                            )

                    traj_seed_arr = np.array([tseed] * len(finl_traj))
                    np.save(
                            os.path.join(savepath, 'trajectory_seed.npy'),
                            traj_seed_arr
                            )

                    health_arr = np.array([h] * len(finl_traj))
                    np.save(
                            os.path.join(savepath, 'health_level.npy'),
                            health_arr
                            )

                    if not only_labels:

                        if i == 0 and tseed not in visited:
                            plot_trajs(base_traj, sick_traj, uniq_traj, finl_traj, saveroot, tseed, s)
                            visited += [tseed]
                            ims = walk2im(finl_traj, image=em, size=size, imsize=imsize)
                        
                        os.makedirs(savepath, exist_ok=True)
                        
                        indices = np.array_split(np.arange(len(ims)), 100)
                        
                        #P.map(save_fn, [(ims, indices[i], savepath) for i in range(len(indices))])
                        for idx, im in enumerate(ims):
                            ssavepath = os.path.join(savepath, f'{idx:0>3d}.png')
                            im.save(ssavepath)

                        impat = os.path.join(savepath, '%03d.png')
                        os.system(f'ffmpeg -y -i {impat} {outfile} > /dev/null 2>&1')

                    Q.put(1)


if __name__ == '__main__':
    make_dataset(only_labels=True)
