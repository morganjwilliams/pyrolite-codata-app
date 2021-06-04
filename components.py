from io import BytesIO, StringIO
import base64
import ipywidgets as widgets
from ipywidgets import HTML
from IPython.display import display
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyrolite.geochem
from pyrolite.plot.color import process_color
from pyrolite.util.plot.legend import proxy_line
from pyrolite.util.plot.style import mappable_from_values
from pyrolite.plot.color import get_cmode

from pyrolite.util.synthetic import normal_frame

# HEADINGS AND DOCS #####################################################################################

heading = widgets.HTML(
    """
<h1>pyrolite Compositional Data Transformer</h1>
<div>
<p>
This is a prototype application for allowing quick and easy point-and-click access
to <a href="https://pyrolite.readthedocs.io"><code>pyrolite</code></a>'s compositional
data functionality.
An example dataset can be generated here, or you can upload your own <code>.csv</code>,
<code>.xls</code> or <code>.xlsx</code> files to work with. Note that this currently will
only work for geochemical data with column names being elements or oxides (e.g. 'Al' or 'Al2O3',
but not 'Al_ppm' or 'Al2O3_wt%'). Here I've also included some basic plotting functionality
from <code>pyrolite</code> so that you can quickly explore the structure in your dataset
and examine how compositional data transformations might be useful.

Your transformed dataset can be exported in the same format you started with
so that you can explore elsewhere, depending on whatever you might be comfortable
using.
</p>
</div>
<hr>
"""
)

# FILE EXPORT ###########################################################################################
exportbuttonhtml = """
<html>
<head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body>
<a download="{filename}" href="data:text/csv;base64,{payload}" download>
<button class="p-Widget jupyter-widgets jupyter-button widget-button mod-success">{buttonname}</button>
</a>
</body>
</html>
"""


def get_export_button(df, filename):
    output_buffer = StringIO()
    extension = Path(filename).suffix
    if extension == ".csv":
        df.to_csv(output_buffer)
    elif extension in [".xlsx", ".xls"]:
        df.to_excel(output_buffer)
    else:
        raise NotImplementedError
    return widgets.HTML(
        exportbuttonhtml.format(
            payload=base64.b64encode(output_buffer.getvalue().encode()).decode(),
            filename=filename,
            buttonname=filename,
        )
    )


# TABS #####################################################################################################


def get_tabs(base):
    return [(base.get_title(ix), tab) for (ix, tab) in enumerate(base.children)]


def add_tabs(base, *tabtuples):
    # tabtuples is a list of tuples = [(title, tab)]
    if tabtuples:
        initial_titles = [base.get_title(ix) for ix in range(len(base.children))]
        base.children = [*base.children, *[t[1] for t in tabtuples]]
        [
            base.set_title(ix, title)
            for ix, title in enumerate(initial_titles + [t[0] for t in tabtuples])
        ]
    return base


def remove_tabs(base, *tabtuples):
    # tabtuples is a list of tuples = [(title, tab)]
    if tabtuples:
        base.children = [c for c in base.children if c not in [t[1] for t in tabtuples]]
    return base


# DATA FUNCTIONS #############################################################################################
def process_frame(df, transform=None):
    non_comp = [c for c in df.columns if c not in df.pyrochem.list_compositional]
    out = df.loc[:, non_comp + df.pyrochem.list_compositional]
    if transform is not None:
        comptransform = getattr(df.pyrochem.compositional.pyrocomp, transform)()
        out = out.join(comptransform)
    return out


def _import_frame(filename, content):
    extension = Path(filename).suffix
    buffer = BytesIO(content)
    if extension == ".csv":
        df = pd.read_csv(buffer)
    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(buffer)
    else:
        raise NotImplementedError(extension)
    df.columns = [c.strip() for c in df.columns]
    df.dropna(how="all")  # drop all empty columns and rows
    df.pyrochem.compositional = df.pyrochem.compositional.apply(
        pd.to_numeric, axis=1, errors="coerce"
    )
    return df


def get_example_dataframe():
    return normal_frame(size=1000)


# TABBED WINDOW ################################################################################################
class Tab:
    def __init__(self, parent, df, name, figsize=(10, 6)):
        self.parent = parent
        self.df = df
        self.name = name
        self.extension = Path(name).suffix
        self.table = widgets.Output()  # datframe output
        self.download_box = widgets.HBox()  # add to children

        # Plotting
        self.plotmode = widgets.RadioButtons(
            options=["xy", "xyz", "ternary"], value="xy"
        )

        self.xvar = widgets.Dropdown(options=[None], value=None)
        self.yvar = widgets.Dropdown(options=[None], value=None)
        self.zvar = widgets.Dropdown(options=[None], value=None)
        self.color = widgets.Dropdown(options=[None], value=None)

        self.logscale_header = widgets.Label("Log Scaling")
        self.logx = widgets.Checkbox(value=False, description="X")
        self.logy = widgets.Checkbox(value=False, description="Y")
        self.logz = widgets.Checkbox(value=False, description="Z")

        self.figure = widgets.Output()

        self.plot_controls = widgets.VBox()

        self.plotbox = widgets.HBox(
            [
                widgets.VBox([self.plotmode, self.plot_controls]),
                self.figure,
            ]
        )

        self.plotmode.observe(self.plotmode_changed, names="value")

        self.parent.transform_dropdown.observe(self.transform_changed, names="value")
        for callback in [
            self.contents_changed,
            self.transform_changed,
            self.plotconfig_changed,
        ]:
            self.parent.uploader.observe(callback, names="value")
        for p in [
            self.xvar,
            self.yvar,
            self.zvar,
            self.color,
            self.logx,
            self.logy,
            self.logz,
        ]:
            p.observe(self.plotconfig_changed, names="value")

        self.box = add_tabs(
            widgets.Accordion(),
            ("Plot", self.plotbox),
            ("Data", widgets.VBox([self.download_box, self.table])),
        )
        self.box.selected_index = 1

        self.contents_changed({})
        self.plotmode_changed({})

    def get_tfm_df(self):
        if self.parent.transform_dropdown.value:
            df = process_frame(self.df, transform=self.parent.transform_dropdown.value)
        else:
            df = self.df
        return df

    def build_export_button(self):
        button = get_export_button(
            self.get_tfm_df(),
            Path(self.name).stem
            + "_"
            + self.parent.transform_dropdown.value
            + "."
            + self.extension,
        )
        box = widgets.HBox([widgets.Label("Download: "), button])
        return box

    def contents_changed(self, change):
        self.transform_changed(change)

    def transform_changed(self, change):
        self.table.clear_output()
        with self.table:
            display(self.get_tfm_df())

        if self.parent.transform_dropdown.value:
            export_button = self.build_export_button()
            self.download_box.children = [export_button]
        else:
            self.download_box.children = ()

        self.update_plotcontrol_options()
        self.plot()

    def plotconfig_changed(self, change):
        self.plot()

    def plotmode_changed(self, change):
        if self.plotmode.value == "xy":
            self.add_2d_plot_controls()
            self.zvar.value = None
        else:
            self.add_3d_plot_controls()
        self.plot()

    def add_2d_plot_controls(self):
        children = [
            widgets.Label("X:"),
            self.xvar,
            widgets.Label("Y:"),
            self.yvar,
            widgets.Label("Color:"),
            self.color,
            widgets.VBox([self.logscale_header, widgets.VBox([self.logx, self.logy])]),
        ]

        self.plot_controls.children = children

    def add_3d_plot_controls(self):
        chlidren = [
            widgets.Label("X:"),
            self.xvar,
            widgets.Label("Y:"),
            self.yvar,
            widgets.Label("Z:"),
            self.zvar,
            widgets.Label("Color:"),
            self.color,
        ]
        """ # Log scales on 3D axes seem to be broken...
        if self.plotmode.value != "ternary":
            chlidren += [
                widgets.VBox(
                    [
                        self.logscale_header,
                        widgets.VBox([self.logx, self.logy, self.logz]),
                    ]
                ),
            ]
        """
        self.plot_controls.children = chlidren

    def update_plotcontrol_options(self):
        _x, _y, _z, _color = (
            self.xvar.value,
            self.yvar.value,
            self.zvar.value,
            self.color.value,
        )
        df = self.get_tfm_df()
        with self.plotbox.hold_sync():  # not sure if this works..
            (
                self.xvar.options,
                self.yvar.options,
                self.zvar.options,
                self.color.options,
            ) = (
                (None, *df.select_dtypes("float").columns),
                (None, *df.select_dtypes("float").columns),
                (None, *df.select_dtypes("float").columns),
                (None, *df.columns),
            )
            for start, var in zip(
                [_x, _y, _z, _color], [self.xvar, self.yvar, self.zvar, self.color]
            ):
                if start in var.options:
                    if not var.value == start:
                        var.value = start

    def plot(self, max_legend_length=12, figsize=(10, 6)):
        # could be cleaner to make a new box and swap childen?
        self.figure.clear_output()
        with self.figure:
            plotvars = [self.xvar.value, self.yvar.value]

            if self.plotmode.value != "xy":
                plotvars += [self.zvar.value]

            plt.close()

            if all([(v is not None) for v in plotvars]):
                naming = [*plotvars]
                if self.color.value is not None:
                    naming.append(self.color.value)

                fig = plt.figure(figsize=figsize, num="-".join(naming))
                frame = self.get_tfm_df()
                c = self.color.value
                if self.plotmode.value != "xyz":
                    ax = fig.add_subplot()
                    ax = frame.loc[:, plotvars].pyroplot.scatter(
                        ax=ax,
                        c=None if c is None else frame[c],
                    )
                    if self.plotmode.value != "ternary":
                        ax.set(
                            xscale=["linear", "log"][self.logx.value],
                            yscale=["linear", "log"][self.logy.value],
                        )
                else:
                    ax = fig.add_subplot(projection="3d")
                    ax.scatter(
                        *frame.loc[:, plotvars].values.T,
                        c=None if c is None else process_color(c=frame[c])["c"]
                    )

                if c is not None:
                    cvals = frame[c]
                    if get_cmode(cvals) == "categories":
                        u = cvals.unique()
                        if len(u) < max_legend_length * 3:
                            proxies = {
                                k: proxy_line(marker="D", lw=0, color=c)
                                for (k, c) in zip(u, process_color(c=u)["c"])
                            }
                            ax.legend(
                                proxies.values(),
                                proxies.keys(),
                                fontsize=14,
                                markerscale=1.5,
                                title=self.color.value,
                                title_fontsize=16,
                                ncol=np.ceil(len(u) / max_legend_length).astype(int),
                            )
                    elif get_cmode(cvals) == "value_array":
                        ax.figure.colorbar(
                            mappable_from_values(cvals),
                            ax=ax,
                            label=self.color.value,
                        )
                    else:
                        pass
                fig.tight_layout()


class MainWindow:
    def __init__(self, valid_extensions=".csv;.xls;.xlsx"):
        self.uploader = widgets.FileUpload(  # note that the counter for file upload is broken, but should be fixed for ipywidgets 8.0
            accept=valid_extensions,  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
            multiple=True,  # True to accept multiple files upload else False
            name="Upload",
        )

        self.transform_dropdown = widgets.Dropdown(
            options=[
                (" - ", ""),
                ("Additive Log-Ratio", "ALR"),
                ("Centred Log-Ratio", "CLR"),
                ("Isometric Log-Ratio", "ILR"),
                ("Spherical Coordinates", "sphere"),
            ],
            value="",
        )
        self.tabs = widgets.Tab()
        self.uploader.observe(self.contents_changed, names="value")
        self.loadexample = widgets.Button(
            description="Load Example Data", button_style="success", icon="download"
        )
        self.loadexample.on_click(self.load_example_data)

        self.box = widgets.VBox(
            [
                heading,
                widgets.HBox(
                    [
                        widgets.Label("Upload CSV file:"),
                        self.uploader,
                        widgets.Label("Select Transform:"),
                        self.transform_dropdown,
                        widgets.Label(" "),
                        self.loadexample,
                    ]
                ),
                self.tabs,
            ]
        )

    def get_contents(self):
        return [
            (fn, _import_frame(fn, d["content"]))
            for (fn, d) in self.uploader.value.items()
        ]

    def contents_changed(self, change):
        self.construct_tabs()

    def load_example_data(self, b):
        self.construct_tabs(example=True)

    def construct_tabs(self, example=False):
        remove_tabs(self.tabs, *get_tabs(self.tabs))  # clear the tabs
        contents = (
            [("Example Data", get_example_dataframe())]
            if example
            else self.get_contents()
        )
        # construct new tabs
        self._tabs = [Tab(self, df, name) for (name, df) in contents]
        add_tabs(self.tabs, *[(t.name, t.box) for t in self._tabs])
