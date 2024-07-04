# Run this app with 'python app.py' and
# visit http://127.0.0.1:8050

from os.path import join
import copy

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import callback_context

from utils import (render, load_data_sample, load_data, config_explanation, config_figure, render_dashboard)
from caching import from_cache

'''
immune data
'''
WORK_FOLDER = '../data/immune'
DATA_SET_NAME = 'Human Immune'
DATA_FILE = 'Human Immune_data.csv'
EMBS_FILE = 'Human Immune_embs.pkl'
# paths to data and embeddings
path_to_data_file = join(WORK_FOLDER, DATA_FILE)
path_to_embs_file = join(WORK_FOLDER, EMBS_FILE)
df_data, df_data_scaled, len_binary = load_data(path_to_data_file)
EMB_NAME = "PCA"

# '''
# UCI adult data
# '''
# WORK_FOLDER = '../data/uci adult'
# DATA_SET_NAME = 'UCI data'
# DATA_FILE = 'adult_clean.csv'
# EMBS_FILE = 'uci_adult_emb.pkl'
# # paths to data and embeddings
# path_to_data_file = join(WORK_FOLDER, DATA_FILE)
# path_to_embs_file = join(WORK_FOLDER, EMBS_FILE)
# df_data, df_data_scaled, len_binary = load_data_sample(path_to_data_file)
# EMB_NAME = "Y"


app, optimiser = render(df_data, df_data_scaled, len_binary, path_to_embs_file, DATA_SET_NAME, EMB_NAME, WORK_FOLDER)


@app.callback(
    Output('explanation', 'children'),
    Input('cluster-select', 'value'),
    prevent_initial_call=True
)
def select_cluster_explanation(cluster):
    labels, attributes, si = optimiser.get_opt_values()
    ics = optimiser.get_ic_opt()
    priors = optimiser.get_priors()
    dls = optimiser.get_dls()
    return config_explanation(df_data, labels, attributes, priors, dls, ics, cluster=cluster)


@app.callback(
    [Output("scatter", "figure"),
     Output("dashboard", "children"),
     Output("cluster-select", "options"),
     Output("cluster-select", "value")],
    [Input("recalc-hyperparameters", "n_clicks"),
     Input("refine-hyperparameters", "n_clicks")],
    [State("alpha-slider", "value"),
     State("beta-slider", "value"),
     State("runtime-slider", "value")],
    prevent_initial_call=True
)
def optimise_different_hyperparameters(change_hyper, refine_hyper, alpha, beta, runtime):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        old_clustering, old_attributes, old_si = optimiser.get_opt_values()
        old_ic = optimiser.get_total_ic_opt()
        if button_id == "recalc-hyperparameters":
            clustering, attributes, si = optimiser.optimise(alpha, beta, runtime)
        else:
            clustering, attributes, si = optimiser.refine(alpha, beta, runtime)
        new_ic = optimiser.get_total_ic_opt()
        options = [{'label': "Cluster " + str(i), 'value': i} for i in list(set(clustering))]
        return config_figure(optimiser.embedding, clustering), \
               render_dashboard(list(set(clustering)), attributes, new_ic, cluster_ids_old=list(set(old_clustering)),
                                attributes_old=old_attributes, ic_old=old_ic), \
               options, \
               0


if __name__ == '__main__':
    app.run_server(debug=True)