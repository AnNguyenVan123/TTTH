import json
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import display
from PIL import Image

file_path = 'experiment_results.json'

if not os.path.exists(file_path):
    raise FileNotFoundError("Không tìm thấy experiment_results.json")

with open(file_path, 'r') as f:
    experiment_data = json.load(f)

os.makedirs("plots", exist_ok=True)

def plot_graph_results(graph_data, idx):
    name = graph_data['graph_name']
    algos = graph_data['algorithms']

    labels = list(algos.keys())
    costs = [algos[a]['cost'] for a in labels]
    times = [algos[a]['time'] for a in labels]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Performance Analysis: {name}", fontsize=16, fontweight='bold')

    ax1 = fig.add_subplot(2, 1, 1)
    for algo, data in algos.items():
        hist = data.get('history', [])
        if len(hist) == 0:
            continue
        x = np.linspace(0, 100, len(hist))
        ax1.plot(x, hist, linewidth=2, label=f"{algo} (Best: {data['cost']})")

    ax1.set_title("Convergence Speed")
    ax1.set_xlabel("Progress (%)")
    ax1.set_ylabel("Objective Value")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2_time = ax2.twinx()

    x_pos = np.arange(len(labels))
    width = 0.35

    ax2.bar(x_pos - width/2, costs, width, label="Cost")
    ax2_time.bar(x_pos + width/2, times, width, label="Time (s)", alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_title("Cost vs Time")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_time.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 🔥 SAVE → LOAD → DISPLAY (KHÔNG PHỤ THUỘC BACKEND)
    img_path = f"plots/plot_{idx}.png"
    plt.savefig(img_path, dpi=150)
    plt.close(fig)

    display(Image.open(img_path))


for i, graph in enumerate(experiment_data):
    plot_graph_results(graph, i)
