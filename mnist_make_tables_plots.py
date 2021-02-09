# %%
import os
import yaml
import itertools
import collections
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy import stats

from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams["text.usetex"] = True
# %%


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def walklevel(some_dir, level=1, equal=False):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir), f"{some_dir} is not a dir"
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        num_sep_this = root.count(os.path.sep)
        if num_sep + level == num_sep_this:
            yield root


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


# daframe helpers
def remove_useless_columns(data, preserve=[]):
    data = data.copy()
    for col in data:
        if all_equal(data.loc[:, col]) and col not in preserve:
            data.pop(col)
    return data


def load_configs(paths):
    """ load all experiments in path folder """

    opts = []
    runs = []
    paths = paths if isinstance(paths, list) else [paths]
    for path in paths:
        runs += [x[0] for x in walklevel(path, level=2)][1:]

    for run in runs:
        if not os.path.isdir(os.path.join(run, ".hydra")):
            continue
        if os.path.isfile(os.path.join(run, "success.txt")):
            with open(os.path.join(run, ".hydra/config.yaml"), "r") as stream:
                args = yaml.safe_load(stream)
            # losses = torch.load(run + "/losses.rar")
            opts.append({"path": run, **args})
    return pd.DataFrame(opts)


def load_experiments(runs, metrics_file="logs/metrics.csv"):
    dfs = []
    for run in runs:
        if not isinstance(metrics_file, list):
            metrics_file = [metrics_file]
        metrics = pd.concat(
            [pd.read_csv(os.path.join(run, f)) for f in metrics_file], axis=0
        ).reset_index(drop=True)
        with open(os.path.join(run, ".hydra/config.yaml"), "r") as stream:
            config = yaml.safe_load(stream)
        config = pd.DataFrame(pd.Series(flatten(config))).transpose()
        config_rep = pd.DataFrame(np.repeat(config.values, metrics.shape[0], axis=0))
        config_rep.columns = config.columns
        run_df = pd.concat([config_rep, metrics], axis=1)
        dfs.append(run_df)
    # print(dfs)
    return pd.concat(dfs, axis=0)


def load_configs_from_name(experiment_names):
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parent = os.path.join(parent, "example_hydra_lightning/experiments")
    paths = list(map(lambda x: os.path.join(parent, x), experiment_names))
    loaded_data = load_configs(paths)
    return loaded_data


# confidence interval helpers
def parse_agg(series):
    return float(series) if len(series) == 1 else series.tolist()


def inter_ci(series, ci=None, stds=None):
    return confidence_interval(series, ci=ci, stds=stds)[1]


def lower_ci(series, ci=None, stds=None):
    return confidence_interval(series, ci=ci, stds=stds)[2]


def upper_ci(series, ci=None, stds=None):
    return confidence_interval(series, ci=ci, stds=stds)[3]


def confidence_interval(series, ci=None, stds=None):
    arr = pd.DataFrame(
        [item] if not isinstance(item, list) else item for item in series
    )

    if (ci is not None) and (stds is not None):
        raise ValueError("One of ci or stds must be none.")
    elif stds is not None:
        count = arr.count(0)
        mean = arr.mean(0)
        std = arr.std(0, ddof=1)
        return tuple(map(parse_agg, (mean, std, mean - std, mean + std)))
    else:
        if ci is None:
            ci = 95
        count = arr.count(0)
        mean = arr.mean(0)
        std = arr.std(0, ddof=1)
        # alpha = 0.05.
        df = len(arr) - 1
        # t = stats.t.ppf(1 - alpha / 2, df)
        t = stats.t.ppf(ci / 2, df)
        sigma = t * std / np.sqrt(count)
        return tuple(map(parse_agg, (mean, sigma, mean - sigma, mean + sigma)))


def filter_data(data, keys, preserve=[]):
    dropna_idx = data[keys].notna().all(axis=1)
    data_filtered = data[dropna_idx]
    data_filtered = remove_useless_columns(data_filtered, preserve)
    # data_filtered = data_filtered.dropna(axis=1)
    return data_filtered


def generate_line_plot(
    data,
    x_key,
    y_key,
    sharex,
    sharey,
    ylim,
    style_first=False,
    legend_outside=True,
    fig_size=None,
    color_index=0,
    style_index=1,
    col_index=2,
    row_index=3,
):
    colors = ["C{}".format(i) for i in range(10)] + ["C{}".format(i) for i in range(10)]
    linestyles = ["solid", (0, (5, 5)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    fontsize = 15

    # Build out dummy index columns up to 4 levels of MultiIndex
    if not isinstance(data.index, pd.MultiIndex):
        n = 1
        index_keys = [data.index.name]
    else:
        n = data.index.nlevels
        index_keys = list(data.index.names)

    for i in range(n, 4):
        data[f"dummy_{i}"] = 1.0
        index_keys.append(f"dummy_{i}")

    data = data.reset_index().set_index(keys=index_keys)

    # Number of rows and cols
    cols = len(data.index.unique(level=2 if style_first else 1))
    rows = len(data.index.unique(level=3))

    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=sharex,
        sharey=sharey,
        figsize=(7 * 1, 6 * rows / cols) if fig_size is None else fig_size,
        squeeze=False,
    )

    # Grab the name of each thing we're grouping by
    color_title = index_keys[0]
    style_title = index_keys[1 if style_first else 2]
    col_title = index_keys[2 if style_first else 1]
    row_title = index_keys[3]

    # Produce a mapping from items in the index to an integer
    # Ensures consistency accross plots even when each index item isn't
    # in each.
    color_mapping = {
        c: i for (i, c) in enumerate(data.index.get_level_values(level=0).unique())
    }
    style_mapping = {
        c: i
        for (i, c) in enumerate(
            data.index.get_level_values(level=1 if style_first else 2).unique()
        )
    }
    col_mapping = {
        c: i
        for (i, c) in enumerate(
            data.index.get_level_values(level=2 if style_first else 1).unique()
        )
    }
    row_mapping = {
        c: i for (i, c) in enumerate(data.index.get_level_values(level=3).unique())
    }

    # Loop over the MultiIndices
    for l1, (index1, df) in enumerate(data.groupby(level=0)):
        for l2, (index2, df) in enumerate(df.groupby(level=1)):
            for l3, (index3, df) in enumerate(df.groupby(level=2)):
                for l4, (index4, df) in enumerate(df.groupby(level=3)):
                    # Grab the name of the index item that maps to each property
                    color_name = index1
                    style_name = index2 if style_first else index3
                    col_name = index3 if style_first else index2
                    row_name = index4

                    # Grab the integer index of each index item
                    color = color_mapping[color_name]
                    style = style_mapping[style_name]
                    col = col_mapping[col_name]
                    row = row_mapping[row_name]

                    # Pull out the right axis
                    ax = axes[row][col]

                    # Set the x/y labels if first row / col
                    if col == 0:
                        ax.set_ylabel(y_key, fontsize=fontsize)

                    if row == 0:
                        ax.set_xlabel(x_key, fontsize=fontsize)

                    # Set axis title to contain the things it is splitting on
                    if n == 4:
                        ax.set_title(
                            f"{col_title}: {col_name}, {row_title}: {row_name}",
                            fontsize=fontsize,
                        )
                    elif (n == 3 and style_first) or (n >= 2 and not style_first):
                        ax.set_title(f"{col_title}: {col_name}", fontsize=fontsize)

                    x = df[x_key]
                    y, y_low, y_up, count = (
                        np.array(df[(y_key, "mean")]),
                        np.array(df[(y_key, "lower_ci")]),
                        np.array(df[(y_key, "upper_ci")]),
                        np.array(df[(y_key, "count")]),
                    )
                    # Set line label to contain the things it is splitting on
                    if (n >= 2 and style_first) or (n >= 3 and not style_first):
                        label = (
                            f"{color_title}: {color_name}, {style_title}: {style_name}"
                        )
                    else:
                        label = f"{color_title}: {color_name}"

                    ax.plot(
                        x,
                        y,
                        lw=2,
                        color=colors[color],
                        linestyle=linestyles[style],
                        label=label,
                    )
                    ax.fill_between(x, y_low, y_up, alpha=0.2, facecolor=colors[color])

    if legend_outside:
        legend = {}
        # Loop through all lines in plots and grab a handle for the legend
        # Avoids duplicate items
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            for (h, l) in zip(handles, labels):
                legend[l] = h
        fig.legend(
            list(legend.values()),
            list(legend.keys()),
            loc="upper left",
            bbox_to_anchor=(1, 0.95),
        )

    for ax in [a for axs in axes for a in axs]:
        if not legend_outside:
            ax.legend(fontsize=4 / 5 * fontsize)
        ax.grid(True)
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])

    plt.tight_layout()


def generate_scatter_plot(
    data,
    y_key,
    sharex,
    sharey,
    ylim,
    uncertainty="scatter",
    title=None,
    style_first=False,
    legend_outside=True,
    legend=True,
    fig_size=None,
    color_index=0,
    style_index=1,
    col_index=2,
    row_index=3,
):
    data = data.copy()
    colors = ["C{}".format(i) for i in range(10)] + ["C{}".format(i) for i in range(10)]
    markerstyles = ["o", "D", "s", "+", "x"]
    fontsize = 15

    # Build out dummy index columns up to 4 levels of MultiIndex
    if not isinstance(data.index, pd.MultiIndex):
        n = 1
        index_keys = [data.index.name]
    else:
        n = data.index.nlevels
        index_keys = list(data.index.names)

    for i in range(n, 4):
        data[f"dummy_{i}"] = 1.0
        index_keys.append(f"dummy_{i}")

    data = data.reset_index().set_index(keys=index_keys)

    # Number of rows and cols
    cols = len(data.index.unique(level=2 if style_first else 1))
    rows = len(data.index.unique(level=3))

    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=sharex,
        sharey=sharey,
        figsize=(7 * 1, 6 * rows / cols) if fig_size is None else fig_size,
        squeeze=False,
    )

    if title is not None:
        fig.suptitle(title)

    # Grab the name of each thing we're grouping by
    color_title = index_keys[0]
    style_title = index_keys[1 if style_first else 2]
    col_title = index_keys[2 if style_first else 1]
    row_title = index_keys[3]

    # Produce a mapping from items in the index to an integer
    # Ensures consistency accross plots even when each index item isn't
    # in each.
    color_mapping = {
        c: i for (i, c) in enumerate(data.index.get_level_values(level=0).unique())
    }
    style_mapping = {
        c: i
        for (i, c) in enumerate(
            data.index.get_level_values(level=1 if style_first else 2).unique()
        )
    }
    col_mapping = {
        c: i
        for (i, c) in enumerate(
            data.index.get_level_values(level=2 if style_first else 1).unique()
        )
    }
    row_mapping = {
        c: i for (i, c) in enumerate(data.index.get_level_values(level=3).unique())
    }

    # Loop over the MultiIndices
    for l1, (index1, df) in enumerate(data.groupby(level=0)):
        for l2, (index2, df) in enumerate(df.groupby(level=1)):
            for l3, (index3, df) in enumerate(df.groupby(level=2)):
                for l4, (index4, df) in enumerate(df.groupby(level=3)):
                    # Grab the name of the index item that maps to each property
                    color_name = index1
                    style_name = index2 if style_first else index3
                    col_name = index3 if style_first else index2
                    row_name = index4

                    # Grab the integer index of each index item
                    color = color_mapping[color_name]
                    style = style_mapping[style_name]
                    col = col_mapping[col_name]
                    row = row_mapping[row_name]

                    # Pull out the right axis
                    ax = axes[row][col]

                    # Set the x/y labels if first row / col
                    if col == 0:
                        ax.set_ylabel(y_key, fontsize=fontsize)

                    # if row == 0:
                    #     ax.set_xlabel(x_key, fontsize=fontsize)

                    # Set axis title to contain the things it is splitting on
                    if n == 4:
                        ax.set_title(
                            f"{col_title}: {col_name}, {row_title}: {row_name}",
                            fontsize=fontsize,
                        )
                    elif (n == 3 and style_first) or (n >= 2 and not style_first):
                        ax.set_title(f"{col_title}: {col_name}", fontsize=fontsize)

                    # x = df[x_key]
                    # y, y_low, y_up, count = (
                    #     np.array(df[(y_key, "mean")]),
                    #     np.array(df[(y_key, "lower_ci")]),
                    #     np.array(df[(y_key, "upper_ci")]),
                    #     np.array(df[(y_key, "count")]),
                    # )
                    y = df[y_key]
                    # Set line label to contain the things it is splitting on
                    if (n >= 2 and style_first) or (n >= 3 and not style_first):
                        # label = (
                        #     f"{color_title}: {color_name}, {style_title}: {style_name}"
                        # )
                        label = (
                            f"{color_title}: {color_name}, {style_title}: {style_name}"
                        )
                    else:
                        # label = f"{color_title}: {color_name}"
                        label = f"{color_title}: {color_name}"

                    x = style + (color * len(style_mapping))
                    if uncertainty == "scatter":
                        ax.scatter(
                            [x] * len(y),
                            y,
                            color=colors[color],
                            marker=markerstyles[style],
                            s=100,
                            # label=label,
                            alpha=0.3,
                        )
                    elif uncertainty == "ci-95":
                        ax.errorbar(
                            x,
                            y.mean(),
                            y.std() * 1.96,  # 95% CI given gaussian assumptions
                            color=colors[color],
                            capsize=5,
                        )
                    elif uncertainty == "std":
                        ax.errorbar(
                            x, y.mean(), y.std(), color=colors[color], capsize=5
                        )
                    elif uncertainty is None:
                        pass
                    else:
                        raise ValueError("Unrecognised uncertainty type")
                    ax.scatter(
                        x,
                        y.mean(),
                        color=colors[color],
                        marker=markerstyles[style],
                        s=100,
                        label=label,
                        alpha=1,
                    )
                    # ax.fill_between(x, y_low, y_up, alpha=0.2, facecolor=colors[color])

    if legend_outside and legend:
        legend = {}
        # Loop through all lines in plots and grab a handle for the legend
        # Avoids duplicate items
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            for (h, l) in zip(handles, labels):
                legend[l] = h
        fig.legend(
            list(legend.values()),
            list(legend.keys()),
            # loc="upper left",
            # bbox_to_anchor=(1, 0.95),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
        )

    ticks = np.arange(len(color_mapping) * len(style_mapping))
    if (n >= 2 and style_first) or (n >= 3 and not style_first):
        # tick_labels = [
        #     f"{color_title}: {color_name}\n{style_title}: {style_name}"
        #     for (color_name, style_name) in itertools.product(
        #         color_mapping.keys(), style_mapping.keys()
        #     )
        # ]
        tick_labels = [
            f"{color_name}\n{style_name}"
            for (color_name, style_name) in itertools.product(
                color_mapping.keys(), style_mapping.keys()
            )
        ]
    else:
        # tick_labels = [
        #     f"{color_title}: {color_name}" for color_name in color_mapping.keys()
        # ]
        tick_labels = [f"{color_name}" for color_name in color_mapping.keys()]

    for ax in [a for axs in axes for a in axs]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90)
        if not legend_outside and legend:
            ax.legend(fontsize=4 / 5 * fontsize)
        ax.grid(True)
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])

    plt.tight_layout()


def generate_scatter_plot_2(
    data,
    y_key,
    sharex,
    sharey,
    ylim,
    uncertainty="scatter",
    title=None,
    style_first=False,
    legend_outside=True,
    legend=True,
    legend_kw={"loc": "upper center", "bbox_to_anchor": (0.5, -0.1), "ncol": 4},
    fig_size=None,
    tick_rotation=90,
    colors=["C{}".format(i) for i in range(10)] + ["C{}".format(i) for i in range(10)],
    markerstyles=["o", "D", "s", "+", "x"],
    fontsize=15,
    color_index=0,
    style_index=1,
    col_index=2,
    row_index=3,
):
    data = data.copy()

    # Build out dummy index columns up to 4 levels of MultiIndex
    if not isinstance(data.index, pd.MultiIndex):
        n = 1
        index_keys = [data.index.name]
    else:
        n = data.index.nlevels
        index_keys = list(data.index.names)

    for i in range(n, 4):
        data[f"dummy {i}"] = 1.0
        index_keys.append(f"dummy {i}")

    data = data.reset_index().set_index(keys=index_keys)

    # Number of rows and cols
    cols = len(data.index.unique(level=col_index))
    rows = len(data.index.unique(level=row_index))

    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=sharex,
        sharey=sharey,
        figsize=(7 * 1, 6 * rows / cols) if fig_size is None else fig_size,
        squeeze=False,
    )

    if title is not None:
        fig.suptitle(title)

    # Grab the name of each thing we're grouping by
    color_title = index_keys[color_index]
    style_title = index_keys[style_index]
    col_title = index_keys[col_index]
    row_title = index_keys[row_index]

    # Produce a mapping from items in the index to an integer
    # Ensures consistency accross plots even when each index item isn't
    # in each.
    color_mapping = {
        c: i
        for (i, c) in enumerate(data.index.get_level_values(level=color_index).unique())
    }
    style_mapping = {
        c: i
        for (i, c) in enumerate(data.index.get_level_values(level=style_index).unique())
    }
    col_mapping = {
        c: i
        for (i, c) in enumerate(data.index.get_level_values(level=col_index).unique())
    }
    row_mapping = {
        c: i
        for (i, c) in enumerate(data.index.get_level_values(level=row_index).unique())
    }

    # Loop over the MultiIndices
    for color_id, (color_name, df) in enumerate(
        data.groupby(level=color_index, sort=False)
    ):
        for style_id, (style_name, df) in enumerate(
            df.groupby(level=style_index, sort=False)
        ):
            for col_id, (col_name, df) in enumerate(
                df.groupby(level=col_index, sort=False)
            ):
                for row_id, (row_name, df) in enumerate(
                    df.groupby(level=row_index, sort=False)
                ):
                    # Grab the integer index of each index item
                    color = color_mapping[color_name]
                    style = style_mapping[style_name]
                    col = col_mapping[col_name]
                    row = row_mapping[row_name]

                    # Pull out the right axis
                    ax = axes[row_id][col_id]

                    # Set the x/y labels if first row / col
                    if col_id == 0:
                        ax.set_ylabel(y_key, fontsize=fontsize)

                    # Set axis title to contain the things it is splitting on
                    if (len(col_mapping) > 1) and (len(row_mapping) > 1):
                        ax.set_title(
                            f"{col_title}: {col_name}\n{row_title}: {row_name}",
                            fontsize=fontsize,
                        )
                    elif len(col_mapping) > 1:
                        ax.set_title(f"{col_title}: {col_name}", fontsize=fontsize)
                    elif len(row_mapping) > 1:
                        ax.set_title(f"{row_title}: {row_name}", fontsize=fontsize)

                    if (len(color_mapping) > 1) and (len(style_mapping) > 1):
                        label = (
                            f"{color_title}: {color_name}, {style_title}: {style_name}"
                        )
                    elif len(color_mapping) > 1:
                        label = f"{color_title}: {color_name}"
                    elif len(style_mapping) > 1:
                        label = f"{style_title}: {style_name}"
                    else:
                        label = None

                    y = df[y_key]

                    x = style + (color * len(style_mapping))
                    if uncertainty == "scatter":
                        ax.scatter(
                            [x] * len(y),
                            y,
                            color=colors[color],
                            marker=markerstyles[style],
                            s=100,
                            # label=label,
                            alpha=0.3,
                        )
                    elif uncertainty == "ci-95":
                        ax.errorbar(
                            x,
                            y.mean(),
                            y.std() * 1.96,  # 95% CI given gaussian assumptions
                            color=colors[color],
                            capsize=5,
                        )
                    elif uncertainty == "std":
                        ax.errorbar(
                            x, y.mean(), y.std(), color=colors[color], capsize=5
                        )
                    elif uncertainty is None:
                        pass
                    else:
                        raise ValueError("Unrecognised uncertainty type")
                    ax.scatter(
                        x,
                        y.mean(),
                        color=colors[color],
                        marker=markerstyles[style],
                        s=100,
                        label=label,
                        alpha=1,
                    )
                    # ax.fill_between(x, y_low, y_up, alpha=0.2, facecolor=colors[color])

    if legend_outside and legend:
        legend = {}
        # Loop through all lines in plots and grab a handle for the legend
        # Avoids duplicate items
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            for (h, l) in zip(handles, labels):
                legend[l] = h
        fig.legend(
            list(legend.values()),
            list(legend.keys()),
            # loc="upper left",
            # bbox_to_anchor=(1, 0.95),
            **legend_kw,
        )

    ticks = np.arange(len(color_mapping) * len(style_mapping))
    if (len(color_mapping) > 1) and (len(style_mapping) > 1):
        tick_labels = [
            f"{color_name}\n{style_name}"
            for (color_name, style_name) in itertools.product(
                color_mapping.keys(), style_mapping.keys()
            )
        ]
    elif len(color_mapping) > 1:
        tick_labels = [f"{color_name}" for color_name in color_mapping.keys()]
    elif len(style_mapping) > 1:
        tick_labels = [f"{style_name}" for style_name in style_mapping.keys()]
    else:
        tick_labels = [None]

    for ax in [a for axs in axes for a in axs]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=tick_rotation)
        if not legend_outside and legend:
            ax.legend(fontsize=4 / 5 * fontsize)
        ax.grid(True)
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])

    plt.tight_layout()

    return fig, axes


def generate_heatmap(data_series, colormap="viridis", title=None, figsize=(6, 8)):
    d = data_series.unstack(level=1)
    level_0 = d.index.values
    level_1 = d.columns
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(d.values, origin="lower", cmap=colormap)
    ax.set_yticks(np.arange(len(level_0)))
    ax.set_yticklabels(level_0)
    ax.set_xticks(np.arange(len(level_1)))
    ax.set_xticklabels(level_1)
    ax.set_ylabel(d.index.names[0])
    ax.set_xlabel(d.columns.names[0])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, use_gridspec=True)
    if title is not None:
        ax.set_title(title)


# plotting helpers
def generate_plot(data, name, metrics, sharey, ylim, offset_per):
    colors = ["C{}".format(i) for i in range(10)] + ["C{}".format(i) for i in range(10)]
    linestyles = ["solid", (0, (5, 5)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    fontsize = 15
    # mean_col = (metric, "mean")
    # ci_col = (metric, "inter_ci")
    # count_col = (metric, "count")

    M = len(data.index.unique(level=0))
    fig, axis = plt.subplots(1, M, sharex=False, sharey=sharey, figsize=(7 * M, 6))
    axes = axis if isinstance(axis, np.ndarray) else [axis]
    subplot_values = data.index.get_level_values(level=0).unique()

    for i, (index, new_df) in enumerate(data.groupby(level=0)):
        ax = axes[list(subplot_values).index(index)]
        if i == 0:
            ax.set_ylabel(metrics[1], fontsize=fontsize)
        ax.set_title("{}={}".format(new_df.index.names[0], index), fontsize=fontsize)
        # print(index, new_df)
        for j, (index, new_df) in enumerate(new_df.groupby(level=1)):
            # print(index, new_df)
            # if isinstance(row.loc[mean_col], float) and math.isnan(row.loc[mean_col]):
            # offset = math.floor(offset_per * len(row.loc[mean_col]))
            # y_ci[:offset], y[:offset] = math.nan, math.nan
            x = new_df[metrics[0]]
            y, y_low, y_up, count = (
                np.array(new_df[(metrics[1], "mean")]),
                np.array(new_df[(metrics[1], "lower_ci")]),
                np.array(new_df[(metrics[1], "upper_ci")]),
                np.array(new_df[(metrics[1], "count")]),
            )
            label = str(
                tuple(
                    "{}={}".format(name, value)
                    for name, value in zip(new_df.index.names, new_df.index[0])
                )
            ) + " #{}".format(count[0])
            ax.plot(x, y, lw=2, color=colors[j], linestyle=linestyles[j], label=label)
            ax.fill_between(x, y_low, y_up, alpha=0.2, facecolor=colors[j])
            # if data.index.nlevels > 1:
            # for k, (index, row) in enumerate(new_df.iterrows()):
            # x =

        ax.legend(fontsize=4 / 5 * fontsize)
        ax.grid(True)
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])

    # title = str(experiment_names) + "\n" + str(criteria)
    title = name
    fig.suptitle(title, fontsize=0.75 * fontsize, y=0.05)
    plt.legend()
    plt.show()
    fig_name = "logs/{}.pdf".format(name)
    # fig.savefig(fig_name, bbox_inches="tight", transparent=True)


# latex helpers
def write_data_to_latex(data):
    latex_path = "table.tex"
    filename = os.path.join(os.getcwd(), latex_path)
    data.to_latex(buf=filename, escape=False, multirow=True, column_format="cccccc")


def compute_ci_and_format(data, level=0):
    new_data = pd.DataFrame()
    for _, new_df in data.groupby(level=level):
        new_df.columns = new_df.columns.droplevel(level=0)
        sorted_idx = np.argsort(new_df["mean"])
        is_gap = (
            new_df.iloc[sorted_idx[0]]["upper_ci"]
            < new_df.iloc[sorted_idx[1]]["lower_ci"]
        )
        bolds = [""] * len(sorted_idx)
        if is_gap:
            bolds[sorted_idx[0]] = "\\bm"
        new_col = [
            "${}{{{:.2f}}}_{{\pm {:.2f}}}$".format(bold, mean, sigma)
            for bold, mean, sigma in zip(bolds, new_df["mean"], new_df["inter_ci"])
        ]
        series = pd.DataFrame(new_col, index=new_df.index, dtype=pd.StringDtype())
        new_data = pd.concat([new_data, series])
    return new_data


def compute_ci_format_table(data, reverse_sort=False, dp=3, ci=None, stds=None):
    data = data.copy()
    contents = {
        k: [
            "mean",
            partial(lower_ci, ci=ci, stds=stds),
            partial(upper_ci, ci=ci, stds=stds),
            "count",
        ]
        for k in data.columns
    }

    gb = list(data.index.names)
    data = data.reset_index().groupby(gb).agg(contents)
    # propogate mean into missing CI columns
    data = data.fillna(axis=1, method="ffill")

    for n, d in data.T.groupby(level=list(range(data.T.index.nlevels - 1))):

        d = d.T
        d.columns = d.columns.droplevel(level=list(range(data.T.index.nlevels - 1)))
        sorted_idx = np.argsort(d["mean"].values)
        best_idx = sorted_idx[0] if reverse_sort else sorted_idx[-1]
        in_best_ci = (
            d["lower_ci"] >= d.iloc[best_idx]["upper_ci"]
            if reverse_sort
            else d["upper_ci"] >= d.iloc[best_idx]["lower_ci"]
        )

        d["mean"] = d["mean"].apply(lambda x: f"{x:0.2f}")
        d["ci"] = (d["upper_ci"] - d["lower_ci"]).apply(lambda x: f"{x:0.2f}")
        d["text"] = "{" + d["mean"] + "$\pm$" + d["ci"] + "}"
        d.loc[in_best_ci, "text"] = "\\textbf" + d[in_best_ci]["text"]
        # d["text"] = "$" + d["text"] + "$"

        data.loc[:, (*n, "text")] = d["text"]

    return data.xs("text", level=data.columns.nlevels - 1, axis=1)


# %%

paths = filter(
    lambda path: ".submitit" not in path,
    itertools.chain(
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/GP-RBF/mnist_experiments_more_chol/batch_size=300,model.chol_noise=1e-06,model.kernel_length_scale=1.0,model.kernel_sigma_var=0.05",
            level=1,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/CNP/mnist_experiments",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/T2-huge/mnist_experiments",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C4-regular_huge/mnist_experiments",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C8-regular_huge/mnist_experiments",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C16-regular_huge/mnist_experiments",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D4-regular_huge/mnist_experiments",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D8-regular_huge/mnist_experiments",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/SO2-irrep_huge/mnist_experiments",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/T2-huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/C4-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/C8-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/C16-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/D4-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/D8-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/SO2-irrep_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/T2-huge/mnist_experiments_blanks",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C4-regular_huge/mnist_experiments_blanks",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C8-regular_huge/mnist_experiments_blanks",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C16-regular_huge/mnist_experiments_blanks",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D4-regular_huge/mnist_experiments_blanks",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D8-regular_huge/mnist_experiments_blanks",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/CNP/mnist_experiments_blanks",
            level=2,
        ),
        # walklevel(
        #     "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/SO2-irrep_huge/mnist_experiments_blanks",
        #     level=2,
        # ),
    ),
)
paths = filter(
    lambda path: os.path.exists(os.path.join(path, "logs", "test_metrics.csv")), paths
)

raw_results = load_experiments(
    paths, metrics_file=["logs/metrics.csv", "logs/test_metrics.csv"]
)
# %%
results = remove_useless_columns(raw_results.copy())
results = results.drop(
    columns=[
        "mean_pred",
        "mean_cov",
        "mean_pred/epoch_0",
        "mean_cov/epoch_0",
        "mean_pred/epoch_0",
        "mean_cov/epoch_0",
        "mean_pred/epoch_1",
        "mean_cov/epoch_1",
        "mean_pred/epoch_2",
        "mean_cov/epoch_2",
        "mean_pred/epoch_3",
        "mean_cov/epoch_3",
        "mean_pred/epoch_4",
        "mean_cov/epoch_4",
        "mean_pred/epoch_5",
        "mean_cov/epoch_5",
        "mean_pred/epoch_6",
        "mean_cov/epoch_6",
        "mean_pred/epoch_7",
        "mean_cov/epoch_7",
        "mean_pred/epoch_8",
        "mean_cov/epoch_8",
        "mean_pred/epoch_9",
        "mean_cov/epoch_9",
    ]
)
results = results.reset_index(drop=True)
# results["model_padding_mode"] = results["model_padding_mode"].fillna("zeros")
# fix gp padding mode for later
# inds = results["model_name"].str.contains("GP")
# results.loc[inds, "model_padding_mode"] = "circular"
# results.loc[inds, "model_cnn_decoder"] = "GP"
# Minor indexing fix
results.loc[
    (results["model_cnn_decoder"] == "C16-regular_huge")
    & (results["dataset_train_args_rotate"] == False)
    & (results["seed"] == 8)
    & (results["model_padding_mode"] == "zeros")
    & results["val_ll"].notna(),
    "step",
] = results.loc[
    (results["model_cnn_decoder"] == "C16-regular_huge")
    & (results["dataset_train_args_rotate"] == False)
    & (results["seed"] == 1)
    & (results["model_padding_mode"] == "zeros")
    & results["val_ll"].notna(),
    "step",
].values
# %%

data = results.copy().rename(
    mapper={
        "dataset_train_args_rotate": "Train dataset",
        "model_cnn_decoder": "Model",
        "val_ll": "validation log likelihood",
        "test_ll": "test log likelihood",
        "train_log_lik": "train log likelihood",
        "no_aug_base": "MNIST",
        "aug_base": "rotMNIST",
        "no_rotate_extrapolate": "extrapolate MNIST",
        "rotate_extrapolate_testset": "extrapolate rotMNIST",
        "model_padding_mode": "Padding mode",
        # "experiment_name": "Finetuned",
    },
    axis=1,
)
data["Model"] = data["Model"].fillna("")
data["Model"] = data["Model"].apply(lambda x: x.split("-")[0])
data.loc[data["model_name"] == "GP-RBF", "Model"] = "GP"
data.loc[data["model_name"] == "CNP", "Model"] = "CNP"
# data = data[~((data.Model == "T2") & (data.seed != 1))]
# data = data[~(data.Model == "SO2")]
# data = data[~((data.Model == "C8") & (data.seed == 17))]
data["Model"] = data["Model"].map(
    {
        "T2": "ConvCNP",
        "C4": "SteerCNP($C_4$)",
        "C8": "SteerCNP($C_{8}$)",
        "C16": "SteerCNP($C_{16}$)",
        "D4": "SteerCNP($D_4$)",
        "D8": "SteerCNP($D_8$)",
        "SO2": "SteerCNP($SO(2)$)",
        "GP": "GP",
        "CNP": "CNP",
    }
)
data["Train dataset"] = data["Train dataset"].apply(
    lambda x: "rotMNIST" if x else "MNIST"
)
data["Type"] = data["experiment_name"].map(
    {
        "mnist_experiments": "Normal",
        "mnist_experiments_finetune": "Finetune",
        "mnist_experiments_more_chol": "Normal",
        "mnist_experiments_blanks": "Blanks",
    }
)


def drop_borked_runs(
    data,
    keys=[
        "test log likelihood",
        "MNIST",
        "rotMNIST",
        "extrapolate MNIST",
        "extrapolate rotMNIST",
        "validation log likelihood",
    ],
    report_cols=["Model", "Padding mode", "seed", "Type"],
    threshold=0,
):
    data = data.copy()
    inds = [data[key] < threshold for key in keys]
    drop_index = inds[0]
    for ind in inds[1:]:
        drop_index = drop_index | ind
    report = data[report_cols][drop_index].drop_duplicates()
    data = data[~drop_index]
    return data, report


data = data.reset_index()
cnp_data = data[data["Model"].str.match("CNP")].copy()
data, report = drop_borked_runs(data[~data["Model"].str.match("CNP")])
data = pd.concat([cnp_data, data])
print(report)

# %%
metrics = ["epoch", "validation log likelihood"]
gb = ["step", "Model", "Train dataset", "Type"]
contents = {
    metrics[0]: [],
    metrics[1]: ["mean", lower_ci, upper_ci, "count"],
}

val_data = filter_data(data.copy(), metrics)
val_data = val_data[(val_data["Type"] == "Blanks")]

val_data = val_data.groupby(gb).agg(contents)
val_data = val_data.reset_index(level=0)

generate_line_plot(
    data=val_data.copy(),
    x_key="step",
    y_key="validation log likelihood",
    sharex=True,
    sharey=True,
    ylim=None,
    style_first=False,
    legend_outside=True,
)
# %%
metrics = ["MNIST", "rotMNIST"]  # , "extrapolate MNIST", "extrapolate rotMNIST"]
# metrics = ["MNIST", "rotMNIST", "extrapolate MNIST", "extrapolate rotMNIST"]
gb = ["Model", "Train dataset", "Type"]
contents = {
    metrics[0]: ["mean", lower_ci, upper_ci, "count"],
}

test_data = data.copy()
# test_data = test_data[test_data["Type"] == "Blanks"]
test_data = test_data[
    ~(
        (test_data["Padding mode"] == "zeros")
        & (test_data["Model"].str.contains("Steer"))
    )
]
test_data = test_data[gb + metrics + ["seed"]]
test_data = remove_useless_columns(test_data)
test_data = test_data.set_index(gb + ["seed"])
test_data = test_data[test_data.notna().any(axis=1)]
test_data = pd.DataFrame(test_data.stack())
test_data.index = test_data.index.set_names(
    [*test_data.index.names[:-1], "Test dataset"]
)
test_data.columns = ["log likelihood"]
test_data.index = test_data.index.droplevel("seed")
test_data = test_data.sort_index(level=1, sort_remaining=False)

test_data = test_data.reset_index()

gp_rows = test_data[test_data["Model"] == "GP"].copy()
gp_rows["Train dataset"] = "rotMNIST"

test_data = pd.concat([gp_rows, test_data])

test_data = test_data.set_index(gb + ["Test dataset"])

generate_scatter_plot_2(
    data=test_data,
    y_key="log likelihood",
    sharex=True,
    sharey=True,
    ylim=None,
    style_first=False,
    legend_outside=True,
    legend=False,
    uncertainty="scatter",
    fig_size=(20, 6),
    color_index=0,
    style_index=2,
    col_index=1,
    row_index=3,
)

plt.savefig("plots/test.pdf")

# %%
metrics = ["MNIST", "rotMNIST"]  # , "extrapolate MNIST", "extrapolate rotMNIST"]
# metrics = ["MNIST", "rotMNIST", "extrapolate MNIST", "extrapolate rotMNIST"]
gb = ["Model", "Train dataset"]
contents = {
    metrics[0]: ["mean", lower_ci, upper_ci, "count"],
}

test_data = data.copy()
test_data = test_data[(test_data["Type"] == "Blanks") | (test_data["Model"] == "GP")]
test_data = test_data[
    ~(
        (test_data["Padding mode"] == "zeros")
        & (test_data["Model"].str.contains("Steer"))
    )
]
test_data = test_data[gb + metrics + ["seed"]]
test_data = remove_useless_columns(test_data)
test_data = test_data.set_index(gb + ["seed"])
test_data = test_data[test_data.notna().any(axis=1)]
test_data = pd.DataFrame(test_data.stack())
test_data.index = test_data.index.set_names(
    [*test_data.index.names[:-1], "Test dataset"]
)
test_data.columns = ["log likelihood"]
test_data.index = test_data.index.droplevel("seed")
test_data = test_data.sort_index(level=1, sort_remaining=False)

test_data = test_data.reset_index()

gp_rows = test_data[test_data["Model"] == "GP"].copy()
gp_rows["Train dataset"] = "rotMNIST"

test_data = pd.concat([gp_rows, test_data])

test_data = test_data.set_index(gb + ["Test dataset"])

fig, axs = generate_scatter_plot_2(
    data=test_data,
    y_key="log likelihood",
    sharex=True,
    sharey=True,
    ylim=None,
    style_first=False,
    legend_outside=True,
    legend=False,
    legend_kw={"loc": "upper center", "bbox_to_anchor": (0.5, -0.05), "ncol": 3},
    uncertainty="std",
    fig_size=(7, 4),
    tick_rotation=-22.5,
    color_index=0,
    style_index=3,
    col_index=2,
    row_index=1,
    fontsize=12,
)

# axs[0][0].set_ylim([0.9, 1.3])

plt.savefig("plots/mnist_train_region.pdf")
# %%
# metrics = ["MNIST", "rotMNIST"]  # , "extrapolate MNIST", "extrapolate rotMNIST"]
metrics = ["extrapolate MNIST", "extrapolate rotMNIST"]
gb = ["Model"]
contents = {
    metrics[0]: ["mean", lower_ci, upper_ci, "count"],
}

test_data = data.copy()
test_data = test_data[(test_data["Type"] == "Blanks") | (test_data["Model"] == "GP")]
test_data = test_data[test_data["Model"] != "CNP"]
test_data = test_data[test_data["Train dataset"] == "MNIST"]
test_data = test_data[gb + metrics + ["seed"]]
test_data = remove_useless_columns(test_data)
test_data = test_data.set_index(gb + ["seed"])
test_data = test_data[test_data.notna().any(axis=1)]
test_data = pd.DataFrame(test_data.stack())
test_data.index = test_data.index.set_names(
    [*test_data.index.names[:-1], "Test dataset"]
)
test_data.columns = ["log likelihood"]
test_data.index = test_data.index.droplevel(1)

test_data = test_data.reset_index()

gp_rows = test_data[test_data["Model"] == "GP"].copy()
gp_rows["Train dataset"] = "rotMNIST"

test_data = pd.concat([gp_rows, test_data])

test_data = test_data.set_index(gb + ["Test dataset"])

fig, axs = generate_scatter_plot_2(
    data=test_data,
    y_key="log likelihood",
    sharex=True,
    sharey=True,
    ylim=None,
    style_first=False,
    legend_outside=True,
    legend=False,
    legend_kw={"loc": "upper center", "bbox_to_anchor": (0.5, -0.05), "ncol": 3},
    uncertainty="scatter",
    fig_size=(7, 2.1),
    tick_rotation=-22.5,
    # colors=['royalblue', 'darkorange', 'forestgreen', 'firebrick', 'mediumorchid', 'gold'],
    color_index=0,
    style_index=3,
    col_index=1,
    row_index=2,
    fontsize=12,
)

# axs[0][0].set_ylim([0.9, 1.3])

plt.savefig("plots/mnist_extrapolation_region.pdf", bbox_inches="tight")
# %%

# metrics = ["MNIST", "rotMNIST"]  # , "extrapolate MNIST", "extrapolate rotMNIST"]
metrics = ["MNIST", "rotMNIST", "extrapolate MNIST", "extrapolate rotMNIST"]
gb = ["Model", "Train dataset"]
# contents = ["mean", partial(lower_ci, stds=1), partial(upper_ci, stds=1), "count"]

test_data = data.copy()
test_data = test_data[(test_data["Type"] == "Blanks") | (test_data["Model"] == "GP")]
test_data = test_data[gb + metrics + ["seed"]]
test_data = remove_useless_columns(test_data)
test_data = test_data.set_index(gb + ["seed"])
test_data = test_data[test_data.notna().any(axis=1)]
test_data = pd.DataFrame(test_data.stack())
test_data.index = test_data.index.set_names(
    [*test_data.index.names[:-1], "Test dataset"]
)
test_data.columns = ["log likelihood"]
# test_data.index = test_data.index.droplevel(2)
test_data = test_data.sort_index(level=1, sort_remaining=False)

test_data = test_data.reset_index()

gp_rows = test_data[test_data["Model"] == "GP"].copy()
gp_rows["Train dataset"] = "rotMNIST"

test_data = pd.concat([gp_rows, test_data])

test_data = test_data.reset_index().pivot_table(
    values="log likelihood",
    columns=["Test dataset", "Train dataset"],
    index=["Model", "seed"],
)
# test_data = test_data.set_index(['Model', 'Train dataset', 'Test dataset', 'seed'])
test_data.index = test_data.index.droplevel("seed")
# test_data = test_data.groupby("Model").agg(contents)
test_data = compute_ci_format_table(test_data, stds=1)
test_data.columns.names = ["Test dataset", "Train dataset"]
test_data = test_data[
    ["MNIST", "rotMNIST", "extrapolate MNIST", "extrapolate rotMNIST"]
]
test_data = test_data.T[
    [
        "GP",
        "CNP",
        "ConvCNP",
        "SteerCNP($C_4$)",
        "SteerCNP($C_{8}$)",
        "SteerCNP($C_{16}$)",
        "SteerCNP($D_4$)",
        "SteerCNP($D_8$)",
    ]
].T

subset_data = test_data[
    [("MNIST", "MNIST"), ("MNIST", "rotMNIST"), ("extrapolate rotMNIST", "MNIST")]
]
subset_data.to_latex(
    buf="plots/mnist_table.tex",
    escape=False,
    multirow=True,
    column_format="l" * subset_data.index.nlevels + "r" * len(subset_data.columns),
)
subset_data
# %%

paths = filter(
    lambda path: ".submitit" not in path,
    itertools.chain(
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/GP-RBF/mnist_experiments_more_chol/",
            level=2,
        ),
    ),
)
paths = filter(
    lambda path: os.path.exists(os.path.join(path, "logs", "test_metrics.csv")), paths
)
# for path in paths:
#     print(path)

gp_results = load_experiments(
    paths, metrics_file=["logs/metrics.csv", "logs/test_metrics.csv"]
)
gp_results = remove_useless_columns(gp_results)
# %%
gp_data = gp_results.copy().rename(
    mapper={
        "dataset_train_args_rotate": "Train dataset",
        # "model_cnn_decoder": "Model",
        # "val_ll": "validation log likelihood",
        # "test_ll": "test log likelihood",
        # "train_log_lik": "train log likelihood",
        "no_aug_base": "MNIST",
        "aug_base": "rotMNIST",
        "no_rotate_extrapolate": "extrapolate MNIST",
        "rotate_extrapolate_testset": "extrapolate rotMNIST",
        "model_kernel_length_scale": "Kernel length scale",
        "model_kernel_sigma_var": "Kernel sigma"
        # "group": "Model",
    },
    axis=1,
)

gp_data = gp_data[gp_data["Kernel length scale"] != 3.0]

# %%

metrics = [
    "MNIST",
    "rotMNIST",
    "extrapolate MNIST",
    "extrapolate rotMNIST",
]
gb = [
    "Kernel length scale",
    "Kernel sigma",
]
contents = {k: ["mean", lower_ci, upper_ci, "count"] for k in metrics}

test_data = gp_data.copy()

test_data = test_data[gb + metrics + ["seed"]].set_index(gb + ["seed"])
test_data = pd.DataFrame(test_data.stack())
test_data.index = test_data.index.set_names(
    [*test_data.index.names[:-1], "Test Dataset"]
)
test_data.columns = ["log likelihood"]
test_data = test_data.unstack()
test_data.columns = test_data.columns.droplevel(level=0)
# test_data = filter_data(data.copy(), metrics)

table_data = test_data.reset_index()[gb + metrics].set_index(gb)

# for m in metrics:
# print(compute_ci_format_table(table_data)[m].unstack(level=1))

compute_ci_format_table(table_data)["MNIST"].unstack(level=1)

# test_data = test_data.reset_index()[gb + metrics].groupby(gb).agg(contents)
# test_data
# %%
plot_data = test_data.reset_index()[gb + metrics].groupby(gb).agg(contents)
generate_heatmap(
    plot_data["MNIST"]["mean"], title=f"Log likelihood on {'MNIST'}", figsize=(3, 3)
)
plt.savefig("plots/mnist_gp_heatmap.pdf")
# %%

paths = filter(
    lambda path: ".submitit" not in path,
    itertools.chain(
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/T2-huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/C4-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/C8-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/C16-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/D4-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/D8-regular_huge/mnist_experiments_finetune",
            level=2,
        ),
        walklevel(
            "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/SO2-irrep_huge/mnist_experiments_finetune",
            level=2,
        ),
    ),
)
paths = filter(
    lambda path: os.path.exists(os.path.join(path, "logs", "metrics.csv")), paths
)

results = load_experiments(paths, metrics_file=["logs/metrics.csv"])
results = remove_useless_columns(results)
# results = results.drop(
#     columns=[
#         "mean_pred",
#         "mean_cov",
#         "mean_pred/epoch_0",
#         "mean_cov/epoch_0",
#         "mean_pred/epoch_0",
#         "mean_cov/epoch_0",
#         "mean_pred/epoch_1",
#         "mean_cov/epoch_1",
#         "mean_pred/epoch_2",
#         "mean_cov/epoch_2",
#         "mean_pred/epoch_3",
#         "mean_cov/epoch_3",
#         "mean_pred/epoch_4",
#         "mean_cov/epoch_4",
#         "mean_pred/epoch_5",
#         "mean_cov/epoch_5",
#         "mean_pred/epoch_6",
#         "mean_cov/epoch_6",
#         "mean_pred/epoch_7",
#         "mean_cov/epoch_7",
#         "mean_pred/epoch_8",
#         "mean_cov/epoch_8",
#         "mean_pred/epoch_9",
#         "mean_cov/epoch_9",
#     ]
# )
results = results.reset_index(drop=True)
results["model_padding_mode"] = results["model_padding_mode"].fillna("zeros")


# %%
data = results.copy().rename(
    mapper={
        "dataset_train_args_rotate": "Train dataset",
        "model_cnn_decoder": "Model",
        "val_ll": "validation log likelihood",
        "test_ll": "test log likelihood",
        "train_log_lik": "train log likelihood",
        "no_aug_base": "MNIST",
        "aug_base": "rotMNIST",
        "no_rotate_extrapolate": "extrapolate MNIST",
        "rotate_extrapolate_testset": "extrapolate rotMNIST",
        "group": "Model",
        "model_padding_mode": "Padding mode",
    },
    axis=1,
)
data["Model"] = data["Model"].apply(lambda x: x.split("-")[0])
data = data[~((data.Model == "T2") & (data.seed != 1))]
data = data[~(data.Model == "SO2")]
data["Model"] = data["Model"].map(
    {
        "T2": "ConvCNP",
        "C4": "SteerCNP($C_4$)",
        "C8": "SteerCNP($C_{8}$)",
        "C16": "SteerCNP($C_{16}$)",
        "D4": "SteerCNP($D_4$)",
        "D8": "SteerCNP($D_8$)",
        "SO2": "SteerCNP($SO(2)$)",
    }
)
data["Train dataset"] = data["Train dataset"].apply(
    lambda x: "rotMNIST" if x else "MNIST"
)
models_index = [
    "ConvCNP",
    "SteerCNP($C_4$)",
    "SteerCNP($C_{8}$)",
    "SteerCNP($C_{16}$)",
    "SteerCNP($D_4$)",
    "SteerCNP($D_8$)",
    "SteerCNP($SO(2)$)",
]

# %%
metrics = ["epoch", "train log likelihood"]
gb = ["step", "Model", "Train dataset", "Padding mode"]
contents = {
    metrics[0]: [],
    metrics[1]: ["mean", lower_ci, upper_ci, "count"],
}

val_data = filter_data(data.copy(), metrics)

val_data = val_data.groupby(gb).agg(contents)
val_data = val_data.reset_index(level=0)

generate_line_plot(
    data=val_data.copy(),
    x_key="step",
    y_key="validation log likelihood",
    sharex=True,
    sharey=True,
    ylim=None,
    style_first=False,
    legend_outside=True,
)
# %%
