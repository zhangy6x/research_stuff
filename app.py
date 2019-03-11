import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash.dependencies import Input, Output

import dash

app = dash.Dash()

external_css = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
]
for css in external_css:
    app.css.append_css({"external_url": css})

layer_name = 'Layer'
ele_name = 'Element'
iso_name = 'Isotope'
iso_ratio_name = 'Isotopic ratio'
iso_tb_rows_default = [{ele_name: None, iso_name: None, iso_ratio_name: None, layer_name: None}]

app.layout = html.Div(
    [
        dcc.Checklist(id='show',
                      options=[
                          {'label': 'Show input field', 'value': True},
                      ], values=[],
                      ),
        html.Div(
            [

                dt.DataTable(rows=iso_tb_rows_default,
                             # columns=header,
                             editable=True,
                             filterable=True,
                             sortable=True,
                             id='table'),
            ],
            id='input_div',
            style={'display': 'none'}
        ),

    ]
)


@app.callback(
    Output('input_div', 'style'),
    [
        Input('show', 'values'),
    ])
def show_hide_iso_table(show):
    if show:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)
