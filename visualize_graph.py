import sys
from typing import Set, Tuple, Union
import networkx as nx
import matplotlib.pyplot as plt

from bert_kg_mvp.utils import parse_triplet_string


def visualize_triplets(triplets: Set[Tuple[str, str, str]]) -> None:
    """Renders a directed NetworkX knowledge graph from a set of (subject, relation, object) tuples."""
    if not triplets:
        print("No valid triplets found to visualize.")
        return

    print(f"Rendering {len(triplets)} edges:")
    for t in triplets:
        print(f"  {t}")

    G = nx.DiGraph()
    for sub, rel, obj in triplets:
        G.add_edge(sub, obj, label=rel)

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=1.0)

    nx.draw_networkx_nodes(G, pos, node_color="#87CEFA", node_size=3000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color="red")

    plt.title("Extracted Knowledge Graph", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def parse_and_visualize(input_data: Union[str, Set[Tuple[str, str, str]]]) -> None:
    """Parses text (if string) and renders the knowledge graph."""
    if isinstance(input_data, str):
        triplets = parse_triplet_string(input_data)
    else:
        triplets = input_data

    visualize_triplets(triplets)


if __name__ == "__main__":
    sample_model_output = sys.argv[1] if len(sys.argv) > 1 else "<triplet> space x <subj_type> entity <relation> located in <obj> california <obj_type> entity"
    parse_and_visualize(sample_model_output)
