import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_1d1_lattice(
    input_vector,
    output_vector=None,
    min_height=None,
    style="standard",
    title=None,
    frame_color="red",
    save_name=None,
    show_xlabel=True,
    show_ylabel=True,
    fig_width_cm=None,  # <-- NEW: Target width in centimeters
    base_fontsize=16,  # <-- NEW: Adjustable font size (defaults to 16 for large plots)
):
    """
    Plots a 1D vector as a 2D binary lattice, or compares two vectors if output_vector is provided.
    """
    in_vec = np.array(input_vector, dtype=int)

    if len(in_vec) == 0:
        print("Input vector is empty.")
        return None

    is_comparison = output_vector is not None

    # --- 1. Data Setup & Padding ---
    if is_comparison:
        out_vec = np.array(output_vector, dtype=int)
        max_len = max(len(in_vec), len(out_vec))

        if len(in_vec) < max_len:
            in_vec = np.pad(in_vec, (0, max_len - len(in_vec)), "constant")
        if len(out_vec) < max_len:
            out_vec = np.pad(out_vec, (0, max_len - len(out_vec)), "constant")

        global_max = max(in_vec.max(), out_vec.max())
        plot_len = max_len
    else:
        global_max = in_vec.max()
        plot_len = len(in_vec)

    height = global_max if min_height is None else max(int(min_height), global_max)
    row_indices = np.arange(height).reshape(-1, 1)

    # --- 2. Image Matrix & Colormap Construction ---
    if is_comparison:
        mask_in = row_indices < in_vec
        mask_out = row_indices < out_vec

        image_matrix = np.zeros((height, plot_len), dtype=int)
        image_matrix[mask_in & mask_out] = 1
        image_matrix[(~mask_in) & mask_out] = 2
        image_matrix[mask_in & (~mask_out)] = 3

        cmap = ListedColormap(["white", "black", "limegreen", "firebrick"])
        vmin, vmax = 0, 3
    else:
        mask = row_indices < in_vec
        image_matrix = np.where(mask, 1, 0)

        cmap = "binary"
        vmin, vmax = None, None

    # --- 3. Figure Initialization ---
    if fig_width_cm is not None:
        # Convert cm to inches for Matplotlib
        cm_to_inch = 1 / 2.54

        # Calculate proportional height to prevent massive white borders
        aspect_ratio = height / plot_len
        fig_height_cm = fig_width_cm * aspect_ratio

        fig, ax = plt.subplots(
            figsize=(fig_width_cm * cm_to_inch, fig_height_cm * cm_to_inch)
        )

    elif style == "naked":
        fig, ax = plt.subplots(figsize=(plot_len / 10, height / 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(
        image_matrix,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_aspect("equal", adjustable="box")

    # Adapt tick sizing for smaller plots
    tick_label_size = max(6, base_fontsize - 2)

    # --- 4. Styling Application ---
    if style == "naked":
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)

        plt.tight_layout(pad=0)

    elif style == "bare_frame":
        # Scale the frame width down slightly if the plot is very small
        frame_lw = 2.0 if (fig_width_cm and fig_width_cm <= 5) else 4.0

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(frame_lw)

        # Conditionally apply labels
        if show_xlabel:
            ax.set_xlabel(
                "Lattice Site Index" if is_comparison else "Site Index",
                fontsize=base_fontsize,
            )
        if show_ylabel:
            ax.set_ylabel("Height / LU", fontsize=base_fontsize)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=tick_label_size,
            width=1.0 if (fig_width_cm and fig_width_cm <= 5) else 2.0,
            length=3.0 if (fig_width_cm and fig_width_cm <= 5) else 6.0,
            direction="out",
            pad=(
                1 if (fig_width_cm and fig_width_cm <= 5) else 8
            ),  # <-- Reduced from 4 to 1
        )

        plt.tight_layout()

    else:  # "standard" style
        if title:
            # Scale title size appropriately
            ax.set_title(title, fontsize=base_fontsize + 2)

        # Conditionally apply labels
        if show_xlabel:
            ax.set_xlabel(
                "Lattice Site Index" if is_comparison else "Site Index",
                fontsize=base_fontsize,
            )
        if show_ylabel:
            ax.set_ylabel("Height / Lattice Units", fontsize=base_fontsize)

        ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

        # Build Legends
        # (For tiny plots, you may need to disable the legend or shrink it further)
        legend_fontsize = max(6, base_fontsize - 4)

        if is_comparison:
            patches = [
                mpatches.Patch(color="black", label="Unchanged"),
                mpatches.Patch(color="limegreen", label="Gain (+)"),
                mpatches.Patch(color="firebrick", label="Loss (-)"),
                mpatches.Patch(facecolor="white", edgecolor="lightgray", label="Empty"),
            ]
            ax.legend(
                handles=patches,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=4,
                frameon=True,
                fontsize=legend_fontsize,
            )
        else:
            patches = [
                mpatches.Patch(color="black", label="atom"),
                mpatches.Patch(facecolor="white", edgecolor="black", label="empty"),
            ]
            ax.legend(
                handles=patches,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                frameon=True,
                fontsize=legend_fontsize,
            )

        # Use minimal padding for small plots to maximize image size
        layout_pad = 0.1 if (fig_width_cm and fig_width_cm <= 5) else 1.0
        plt.tight_layout(pad=layout_pad)

    # --- 5. Save and Display ---
    if save_name:
        fig.savefig(save_name, bbox_inches="tight", pad_inches=0.01)

    plt.show()
    return fig
