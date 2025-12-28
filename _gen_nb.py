
import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Analysis of Marker Vector Evolution\n",
            "\n",
            "This notebook analyzes how the marker vectors (`stream_id_A` and `stream_id_B`) evolve during training for a specific run (`run_20251221_112940`). We inspect the saved checkpoints to trace their trajectory, norms, and relationship."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import torch\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import json\n",
            "from tqdm import tqdm\n",
            "import torch.nn.functional as F\n",
            "from sklearn.decomposition import PCA\n",
            "\n",
            "# Set plotting style\n",
            "plt.style.use('seaborn-v0_8-darkgrid')\n",
            "plt.rcParams['figure.figsize'] = (12, 8)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configuration\n",
            "RUN_DIR = \"checkpoints/run_20251221_112940\"\n",
            "\n",
            "if not os.path.exists(RUN_DIR):\n",
            "    print(f\"Warning: {RUN_DIR} does not exist. Please check the path.\")\n",
            "else:\n",
            "    print(f\"Analyzing run: {RUN_DIR}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_checkpoints(run_dir):\n",
            "    checkpoints = []\n",
            "    if not os.path.exists(run_dir):\n",
            "        return []\n",
            "    for name in os.listdir(run_dir):\n",
            "        if name.startswith(\"checkpoint-\"):\n",
            "            try:\n",
            "                step = int(name.split(\"-\")[1])\n",
            "                path = os.path.join(run_dir, name, \"model.pt\")\n",
            "                if os.path.exists(path):\n",
            "                    checkpoints.append((step, path))\n",
            "            except ValueError:\n",
            "                continue\n",
            "    return sorted(checkpoints, key=lambda x: x[0])\n",
            "\n",
            "checkpoints = get_checkpoints(RUN_DIR)\n",
            "print(f\"Found {len(checkpoints)} checkpoints.\")\n",
            "for step, _ in checkpoints:\n",
            "    print(f\"Step: {step}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "data = {\n",
            "    \"steps\": [],\n",
            "    \"vec_A\": [],\n",
            "    \"vec_B\": []\n",
            "}\n",
            "\n",
            "print(\"Starting extraction...\")\n",
            "\n",
            "for step, path in tqdm(checkpoints, desc=\"Loading checkpoints\"):\n",
            "    try:\n",
            "        # Load checkpoint on CPU to save memory\n",
            "        ckpt = torch.load(path, map_location=\"cpu\")\n",
            "        state_dict = ckpt[\"model_state_dict\"]\n",
            "        \n",
            "        vec_A = None\n",
            "        vec_B = None\n",
            "        \n",
            "        # Handle potential FSDP or module prefixes\n",
            "        keys_A = [\"stream_id_A\", \"module.stream_id_A\", \"_orig_mod.stream_id_A\"]\n",
            "        keys_B = [\"stream_id_B\", \"module.stream_id_B\", \"_orig_mod.stream_id_B\"]\n",
            "        \n",
            "        for k in keys_A:\n",
            "            if k in state_dict:\n",
            "                vec_A = state_dict[k].float().numpy()\n",
            "                break\n",
            "        \n",
            "        for k in keys_B:\n",
            "            if k in state_dict:\n",
            "                vec_B = state_dict[k].float().numpy()\n",
            "                break\n",
            "                \n",
            "        if vec_A is not None and vec_B is not None:\n",
            "            data[\"steps\"].append(step)\n",
            "            data[\"vec_A\"].append(vec_A)\n",
            "            data[\"vec_B\"].append(vec_B)\n",
            "            \n",
            "    except Exception as e:\n",
            "        print(f\"Error loading checkpoint {step}: {e}\")\n",
            "\n",
            "if len(data[\"steps\"]) > 0:\n",
            "    # Reshape: (N, 1, 1, D) -> (N, D) or similar depending on original shape\n",
            "    data[\"vec_A\"] = np.array(data[\"vec_A\"]).reshape(len(data[\"steps\"]), -1)\n",
            "    data[\"vec_B\"] = np.array(data[\"vec_B\"]).reshape(len(data[\"steps\"]), -1)\n",
            "    steps = np.array(data[\"steps\"])\n",
            "\n",
            "    print(f\"Extracted data shape A: {data['vec_A'].shape}\")\n",
            "    print(f\"Extracted data shape B: {data['vec_B'].shape}\")\n",
            "else:\n",
            "    print(\"No data extracted!\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "if len(data[\"steps\"]) > 0:\n",
            "    # 1. Norms\n",
            "    norm_A = np.linalg.norm(data[\"vec_A\"], axis=1)\n",
            "    norm_B = np.linalg.norm(data[\"vec_B\"], axis=1)\n",
            "\n",
            "    # 2. Euclidean Distance\n",
            "    dist_AB = np.linalg.norm(data[\"vec_A\"] - data[\"vec_B\"], axis=1)\n",
            "\n",
            "    # 3. Cosine Similarity\n",
            "    dot_prod = np.sum(data[\"vec_A\"] * data[\"vec_B\"], axis=1)\n",
            "    cos_sim = dot_prod / (norm_A * norm_B)\n",
            "\n",
            "    # 4. Drift from start\n",
            "    init_A = data[\"vec_A\"][0]\n",
            "    init_B = data[\"vec_B\"][0]\n",
            "\n",
            "    drift_A = np.linalg.norm(data[\"vec_A\"] - init_A, axis=1)\n",
            "    drift_B = np.linalg.norm(data[\"vec_B\"] - init_B, axis=1)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "if len(data[\"steps\"]) > 0:\n",
            "    fig, axs = plt.subplots(2, 2, figsize=(15, 12))\n",
            "    fig.suptitle(f'Marker Vector Evolution - {os.path.basename(RUN_DIR)}', fontsize=16)\n",
            "\n",
            "    # Plot 1: Norms\n",
            "    axs[0, 0].plot(steps, norm_A, 'b-o', label='Stream A Marker')\n",
            "    axs[0, 0].plot(steps, norm_B, 'r-o', label='Stream B Marker')\n",
            "    axs[0, 0].set_title('L2 Norm of Marker Vectors')\n",
            "    axs[0, 0].set_xlabel('Step')\n",
            "    axs[0, 0].set_ylabel('L2 Norm')\n",
            "    axs[0, 0].legend()\n",
            "\n",
            "    # Plot 2: Distance between A and B\n",
            "    axs[0, 1].plot(steps, dist_AB, 'g-o', label='Distance (A - B)')\n",
            "    axs[0, 1].set_title('Euclidean Distance between Markers A and B')\n",
            "    axs[0, 1].set_xlabel('Step')\n",
            "    axs[0, 1].set_ylabel('Euclidean Distance')\n",
            "    axs[0, 1].legend()\n",
            "\n",
            "    # Plot 3: Cosine Similarity\n",
            "    axs[1, 0].plot(steps, cos_sim, 'purple', marker='o', label='Cosine Similarity')\n",
            "    axs[1, 0].set_title('Cosine Similarity between Markers A and B')\n",
            "    axs[1, 0].set_xlabel('Step')\n",
            "    axs[1, 0].set_ylabel('Cosine Similarity')\n",
            "    axs[1, 0].set_ylim(-1.1, 1.1)\n",
            "    axs[1, 0].legend()\n",
            "\n",
            "    # Plot 4: Drift from Start\n",
            "    axs[1, 1].plot(steps, drift_A, 'b--', label='Drift A')\n",
            "    axs[1, 1].plot(steps, drift_B, 'r--', label='Drift B')\n",
            "    axs[1, 1].set_title('Distance from First Saved Checkpoint')\n",
            "    axs[1, 1].set_xlabel('Step')\n",
            "    axs[1, 1].set_ylabel('Euclidean Distance')\n",
            "    axs[1, 1].legend()\n",
            "\n",
            "    plt.tight_layout()\n",
            "    plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "if len(data[\"steps\"]) > 0:\n",
            "    # Combine for PCA\n",
            "    all_vecs = np.concatenate([data[\"vec_A\"], data[\"vec_B\"]], axis=0)\n",
            "\n",
            "    pca = PCA(n_components=2)\n",
            "    pca_result = pca.fit_transform(all_vecs)\n",
            "\n",
            "    n = len(steps)\n",
            "    pca_A = pca_result[:n]\n",
            "    pca_B = pca_result[n:]\n",
            "\n",
            "    plt.figure(figsize=(10, 8))\n",
            "\n",
            "    plt.plot(pca_A[:, 0], pca_A[:, 1], 'b-o', label='Trajectory A', alpha=0.6)\n",
            "    plt.plot(pca_B[:, 0], pca_B[:, 1], 'r-o', label='Trajectory B', alpha=0.6)\n",
            "\n",
            "    # Start/End markers\n",
            "    plt.scatter(pca_A[0, 0], pca_A[0, 1], c='blue', s=200, marker='*', label='Start A')\n",
            "    plt.scatter(pca_A[-1, 0], pca_A[-1, 1], c='blue', s=200, marker='X', label='End A')\n",
            "    plt.scatter(pca_B[0, 0], pca_B[0, 1], c='red', s=200, marker='*', label='Start B')\n",
            "    plt.scatter(pca_B[-1, 0], pca_B[-1, 1], c='red', s=200, marker='X', label='End B')\n",
            "\n",
            "    plt.title('2D PCA Projection of Marker Vector Trajectories')\n",
            "    plt.xlabel('PC1')\n",
            "    plt.ylabel('PC2')\n",
            "    plt.legend()\n",
            "    plt.grid(True)\n",
            "    plt.show()"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("analyze_marker_evolution.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully: analyze_marker_evolution.ipynb")




