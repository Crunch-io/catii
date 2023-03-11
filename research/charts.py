import pandas as pd
import seaborn as sns


def gen_chart(
    title,
    data,
    x="rows",
    y="seconds",
    row=None,  # The record field which will be used to generate multiple rows of charts
    col=None,  # The record field which will be used to generate multiple columns of charts
    style="extent",
    hue=None,
    xscale="log",
    yscale="log",
    redlineat1=False,
    ylim=None,
):
    df = pd.DataFrame(data)
    palette = (
        None
        if hue is None
        else dict(zip(df[hue].unique(), sns.color_palette() + [(1.0, 0.0, 0.5)]))
    )

    # dash_styles = [
    #     "",
    #     (4, 1.5),
    #     (1, 1),
    #     (3, 1, 1.5, 1),
    #     (5, 1, 1, 1),
    #     (5, 1, 2, 1, 2, 1),
    #     (2, 2, 3, 1.5),
    #     (1, 2.5, 3, 1.2)
    # ]

    print("Generating '%s' chart" % title)
    sns.set(style="ticks")
    plot = sns.relplot(
        x=x,
        y=y,
        row=row,
        col=col,
        # dashes=dash_styles,
        style=style,
        hue=hue,
        size_order=["T1", "T2"],
        palette=palette,
        height=5,
        aspect=1.0,
        facet_kws=dict(sharex=False),
        kind="line",
        legend="full",
        data=df,
    )
    for row in plot.axes:
        for ax in row:
            ax.set_xscale(xscale)
            if yscale:
                ax.set_yscale(yscale)
            ax.set_xlabel(x)
            if ylim is not None:
                ax.set(ylim=ylim)
    plot.fig.subplots_adjust(top=0.90, bottom=0.1, hspace=0.3)
    plot.fig.suptitle(title)

    if redlineat1:
        plot.refline(y=1.0, color="red")

    plot.savefig("plot-%s.png" % title.replace(" ", "-").replace(":", "_"))
