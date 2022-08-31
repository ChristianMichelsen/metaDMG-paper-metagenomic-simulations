#%%
from pathlib import Path

import numpy as np
import pandas as pd
import parse
import plotly.express as px
import plotly.graph_objects as go

#%%

parser_comparison = parse.compile(
    "{sample}.{N_reads:Int}.{sim_name}.comparison",
    dict(Int=int),
)


def get_sample_N_reads_simulation_method(path):
    d = parser_comparison.parse(path.stem).named
    return d["sample"], d["N_reads"], d["sim_name"]


for path in Path("data/analysis_comparison").glob("*.csv"):
    sample, N_reads, simulation_method = get_sample_N_reads_simulation_method(path)

    df_comparison = pd.read_csv(path)


#%%

#%%

fig_A_B = px.scatter(
    df_comparison,
    x="tax_id",
    y="|B|/|A|",
    hover_data=["tax_name", "tax_id", "|A|", "|B|"],
    title=f"{sample}, {N_reads} reads: |B| / |A|",
)
fig_A_B.show()

#%%

fig_B_C = px.scatter(
    df_comparison,
    x="tax_id",
    y="|C|/|B|",
    hover_data=["tax_name", "tax_id", "|B|", "|C|"],
    title=f"{sample}, {N_reads} reads: |C| / |B|",
)
fig_B_C.show()


#%%


fig_A_C = px.scatter(
    df_comparison,
    x="tax_id",
    y="|C|/|A|",
    hover_data=["tax_name", "tax_id", "|A|", "|C|"],
    title=f"{sample}, {N_reads} reads: |C| / |A|",
)
fig_A_C.show()


#%%


fig_CT_mismatch_vs_C = px.scatter(
    df_comparison,
    x="C) f_CT (x=1)",
    y="mismatch) f_CT (x=1)",
    hover_data=[
        "tax_name",
        "tax_id",
        "C) k_CT (x=1)",
        "C) N_C (x=1)",
        "mismatch) k_CT (x=1)",
        "mismatch) N_C (x=1)",
    ],
    title="C->T: mismatch vs C",
)
fig_CT_mismatch_vs_C.add_trace(px.line(y=[0, 1]).data[0])


fig_CT_mismatch_vs_C.update_xaxes(
    range=[0, 1],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
)

fig_CT_mismatch_vs_C.update_yaxes(
    range=[0, 1],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
)

fig_CT_mismatch_vs_C.show()


#%%


# Build figure
fig_CT_pipeline = go.Figure()

kwargs = dict(
    mode="markers",
    # marker_size=10,
    # marker_line_color="black",
    marker_line_color="lightgrey",
    marker_line_width=0.5,
    showlegend=True,
)


_ = [
    ("A", "A) f_CT (x=1)", "#C00000 ", "circle"),
    ("B", "B) f_CT (x=1)", "#0070C0", "square"),
    ("C", "C) f_CT (x=1)", "#00B050", "diamond"),
    ("A\B", "A\B) f_CT (x=1)", "#802540", "hexagon"),
    ("B\C", "B\C) f_CT (x=1)", "#00859B", "star-square"),
    ("mismatch", "mismatch) f_CT (x=1)", "#FFC000", "star"),
]

for (name, variable, color, symbol) in _:

    customdata = np.stack(
        (
            df_comparison["tax_id"],
            df_comparison["tax_name"],
            df_comparison[f"|{name}|"],
            df_comparison[f"{name}) k_CT (x=1)"],
            df_comparison[f"{name}) N_C (x=1)"],
        )
    ).T

    # Add scatter trace with medium sized markers
    fig_CT_pipeline.add_trace(
        go.Scatter(
            x=df_comparison["|A|"],
            y=100 * df_comparison[variable],
            name=name,
            marker_color=color,
            marker_symbol=symbol,
            marker_size=1 + 2 * np.log(1 + df_comparison[f"|{name}|"]),
            customdata=customdata,
            hovertemplate=""
            + "<br>"
            + "Tax ID = %{customdata[0]}"
            + "<br>"
            + "Tax name = %{customdata[1]}"
            + "<br>"
            + "<br>"
            + f"|{name}| "
            + "= %{customdata[2]}"
            + "<br>"
            + f"{name}) "
            + "k_CT (x=1) = %{customdata[3]}"
            + "<br>"
            + f"{name}) "
            + "N_C (x=1) = %{customdata[4]}"
            + "<br>"
            + f"{name}) "
            + "f_CT (x=1) = %{y:.1f}%"
            + "<br>",
            # + "<extra></extra>",
            **kwargs,
        )
    )

fig_CT_pipeline.update_xaxes(type="log")

fig_CT_pipeline.update_layout(
    title=f"{sample}, {N_reads} reads: f_CT (x=1)",
    xaxis_title="|A|",
    yaxis_title="f_CT (x=1)",
    autosize=True,
    width=1600 / 1.5,
    height=900 / 1.5,
)


fig_CT_pipeline.show()

fig_CT_pipeline.write_html("file.html")

# %%
