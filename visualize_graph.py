import sys
import re
import networkx as nx
import matplotlib.pyplot as plt

def parse_and_visualize(text: str):
    # 1. Parse the string using the robust Regex pattern
    triplets = set()
    pattern = r"<triplet>\s*(.*?)\s*<subj_type>.*?<relation>\s*(.*?)\s*<obj>\s*(.*?)\s*<obj_type>"
    matches = re.findall(pattern, text)
    
    for sub, rel, obj in matches:
        if sub and rel and obj:
            triplets.add((sub.strip(), rel.strip(), obj.strip()))
            
    if not triplets:
        print("No valid triplets found to visualize.")
        return

    print(f"Extracted {len(triplets)} edges:")
    for t in triplets: print(f"  {t}")

    # 2. Build the Directed Graph
    G = nx.DiGraph()
    for sub, rel, obj in triplets:
        G.add_edge(sub, obj, label=rel)

    # 3. Render the Graph
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=1.0) # Spaces out the nodes
    
    nx.draw_networkx_nodes(G, pos, node_color='#87CEFA', node_size=3000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='red')

    plt.title("Extracted Knowledge Graph", pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # You can pass your model's raw string output directly to this script
    sample_model_output = sys.argv[1] if len(sys.argv) > 1 else "<triplet> space x <subj_type> entity <relation> located in <obj> california <obj_type> entity"
    parse_and_visualize(sample_model_output)
