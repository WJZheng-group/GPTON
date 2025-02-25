import obonet
import pandas as pd
import networkx as nx

# Match back result
eval_nodes = pd.read_csv('/data1/jli49/GEN_RESP/old_data/back_trace_2k70b_woshot_all_bp_top_5.csv')

for i in range(5):
    eval_nodes[f'Graph_check_{i+1}_len2'] = 0
    eval_nodes[f'Graph_check_{i+1}_len3'] = 0

# go-basic obo file
dgraph = obonet.read_obo('/data1/jli49/GEN_RESP/go.obo')
graph = dgraph.to_undirected()

name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}

real_gos = eval_nodes['Real_GO'].tolist()
matched_gos_list = [eval_nodes[f'Matched_GO_{i+1}'].tolist() for i in range(5)]

def get_shortest_path_length(graph, node1, node2):
    if node1 in graph and node2 in graph and nx.has_path(graph, node1, node2):
        return nx.shortest_path_length(graph, node1, node2)
    return float('inf')

def graph_check_each_matched(eval_nodes, real_gos, matched_gos, graph, name_to_id, n):
    for i, (real_go, matched_go) in enumerate(zip(real_gos, matched_gos)):
        real_node = name_to_id.get(real_go, None)
        matched_node = name_to_id.get(matched_go, None)
        shortest_path_length = get_shortest_path_length(graph, real_node, matched_node)

        if shortest_path_length <= 2:
            eval_nodes.at[i, f'Graph_check_{n+1}_len2'] = 1
        if shortest_path_length <= 3:
            eval_nodes.at[i, f'Graph_check_{n+1}_len3'] = 1

def graph_check_topn_matched(eval_nodes, n, length):
    col_name = f'Graph_check_top{n}_len{length}'
    eval_nodes[col_name] = 0
    for index, row in eval_nodes.iterrows():
        for i in range(n):
            if row[f'Graph_check_{i+1}_len{length}'] == 1:
                eval_nodes.at[index, col_name] = 1
                break

for n, matched_gos in enumerate(matched_gos_list):
    graph_check_each_matched(eval_nodes, real_gos, matched_gos, graph, name_to_id, n)

graph_check_topn_matched(eval_nodes, 3, 2)
graph_check_topn_matched(eval_nodes, 3, 3)
graph_check_topn_matched(eval_nodes, 5, 2)
graph_check_topn_matched(eval_nodes, 5, 3)

eval_nodes.to_csv('/data1/jli49/GEN_RESP/data/2k70b_woshot_all_bp_top_5_path_len.csv', index=False)
