import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


matplotlib.use('Agg')


def viz_matches(
    img1, kp1, img2, kp2, filepath,
    all_kp1=None, all_kp2=None, color=None,
    kp_size=4, thickness=2.0, margin=20,
    cmap='brg', ylabel='', normalize=True,
    r=(0, 1), dpi=100, title=None
):

    if len(img1.shape) == 2:
        img1 = img1[..., np.newaxis]
    if len(img2.shape) == 2:
        img2 = img2[..., np.newaxis]
    if img1.shape[-1] == 1:
        img1 = np.repeat(img1, 3, -1)
    if img2.shape[-1] == 1:
        img2 = np.repeat(img2, 3, -1)

    tile_shape = (
        img1.shape[0] + img2.shape[0] + margin,
        max(img1.shape[1], img2.shape[1]), img1.shape[2]
    )
    tile = np.ones(tile_shape, type(img1.flat[0]))
    if np.max(img1) > 1 or np.max(img2) > 1:
        tile *= 255
    
    tile[0:img1.shape[0], 0:img1.shape[1]] = img1
    y_thresh1 = img1.shape[0] + margin
    y_thresh2 = img1.shape[0] + img2.shape[0] + margin
    tile[y_thresh1:y_thresh2, 0:img2.shape[1]] = img2

    fig = plt.figure(frameon=False, dpi=dpi)
    w, h, _ = tile.shape
    fig_size = 12
    fig.set_size_inches(fig_size, fig_size / h * w)
    ax = plt.Axes(fig, [0., 0., 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(tile,
              cmap=plt.get_cmap(cmap),
              vmin=None if normalize else r[0],
              vmax=None if normalize else r[1],
              aspect='auto')
    
    kp2[:, 1] += img1.shape[0] + margin
    xs = np.stack([kp1[:, 0], kp2[:, 0]], 0)
    ys = np.stack([kp1[:, 1], kp2[:, 1]], 0)

    if isinstance(color, list):
        for i, c in enumerate(color):
            ax.plot(
                xs[:, i],
                ys[:, i],
                linestyle='-',
                linewidth=thickness,
                aa=True,
                marker='.',
                markersize=kp_size,
                color=c,
            )
    else:
        ax.plot(
            xs,
            ys,
            linestyle='-',
            linewidth=thickness,
            aa=True,
            marker='.',
            markersize=kp_size,
            color=color,
        )
    
    if all_kp1 is not None and all_kp1 is not None:
        all_kp2[:, 1] += img1.shape[0] + margin
        ax.scatter(all_kp1[:, 0],
                   all_kp1[:, 1],
                   s=kp_size-1,
                   c=(0, 1, 0))
        ax.scatter(all_kp2[:, 0],
                   all_kp2[:, 1],
                   s=kp_size-1,
                   c=(0, 1, 0))


    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=None)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=None)
    plt.savefig(filepath, format='jpg')
    plt.close()