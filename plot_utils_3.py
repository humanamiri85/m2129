def plotfunction(Z, Theta, R, coords_df, radii,
                 cmap='jet', title='', cbar_label='',
                 vmin=None, vmax=None, levels=None, nlevels: int = 21,
                 extend: str = 'neither',
                 show_points: bool = True,
                 draw_guides: bool = True,
                 show: bool = True,
                 ax=None,
                 pad=20):

    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Create polar axis if not provided
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 6))
    else:
        # ✅ Safe: Ensure ax is polar
        if not hasattr(ax, 'set_theta_zero_location'):
            raise ValueError("Provided ax is not a polar axis. Use subplot_kw=dict(polar=True).")
        fig = ax.get_figure()

    # 2) Input validation
    if Z is None or np.all(np.isnan(Z)):
        raise ValueError("Z contains only NaN values - cannot plot.")
    if Z.shape != Theta.shape or Z.shape != R.shape:
        raise ValueError(f"Z, Theta, R must have same shape. Got Z: {Z.shape}, Theta: {Theta.shape}, R: {R.shape}")

    # 3) Set levels
    if levels is None:
        if vmin is None: vmin = np.nanmin(Z)
        if vmax is None: vmax = np.nanmax(Z)
        if np.isclose(vmin, vmax, rtol=1e-12):
            vmax = vmin + 1e-12
        levels = np.linspace(vmin, vmax, nlevels)

    # 4) Plot
    try:
        cs = ax.contourf(Theta, R, Z, levels=levels, cmap=cmap, extend=extend)
    except Exception as e:
        raise RuntimeError(f"contourf failed: {e}. Check Z, Theta, R shapes and values.")

    # 5) Title & colorbar
    ax.set_title(title, pad=pad)
    cbar = fig.colorbar(cs, ax=ax, pad=0.1)
    if cbar_label:
        cbar.set_label(cbar_label)

    # 6) Probe points
    if show_points and coords_df is not None and not coords_df.empty:
        ax.scatter(coords_df['theta'], coords_df['r'],
                   s=50, facecolors='none', edgecolors='black', marker='o', zorder=5)

    # 7) Radial limits
    max_r = 1.01 * np.max(radii) if radii is not None and len(radii) > 0 else 1.01 * np.nanmax(coords_df['r'])
    ax.set_ylim(0, max_r)

    # 8) Grid
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # 9) Guides
    if draw_guides:
        if radii is not None:
            th = np.linspace(0, 2*np.pi, 500)
            for radius in radii:
                ax.plot(th, np.full_like(th, radius), linestyle='--', color='gray', linewidth=0.9)
        if coords_df is not None and not coords_df.empty:
            unique_thetas = np.unique(coords_df['theta'])
            for t in unique_thetas:
                r_min = coords_df['r'].min()
                r_max = coords_df['r'].max()
                ax.plot([t, t], [r_min, r_max], linestyle='--', color='gray', linewidth=0.9)

    # 10) Show
    if show:
        plt.show()

    return fig, ax, cs
