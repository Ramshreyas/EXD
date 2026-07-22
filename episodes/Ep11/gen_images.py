"""Generate images for Ep11 article from the notebook cells."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(OUT, exist_ok=True)

cat = np.array([0.8, 0.3])
dog = np.array([0.6, 0.7])
random_vec = np.array([-0.2, 0.9])

# --- Section 1: Vectors in 2D ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.quiver(0, 0, cat[0], cat[1], angles='xy', scale_units='xy', scale=1,
          color='steelblue', width=0.02, label='cat = [0.8, 0.3]')
ax.quiver(0, 0, dog[0], dog[1], angles='xy', scale_units='xy', scale=1,
          color='darkorange', width=0.02, label='dog = [0.6, 0.7]')
ax.set_xlim(-0.1, 1.0); ax.set_ylim(-0.1, 1.0)
ax.set_xlabel('dim 1'); ax.set_ylabel('dim 2')
ax.axhline(y=0, color='gray', alpha=0.3); ax.axvline(x=0, color='gray', alpha=0.3)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
ax.set_title("Word Vectors in 2D", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'vectors_2d.png'), dpi=140, bbox_inches='tight')
plt.close()
print("✓ vectors_2d.png")

# --- Section 2: Rotation ---
angles = np.linspace(0, 2*np.pi, 13)[:-1]
rotated = np.array([
    np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ cat
    for t in angles
])
fig, ax = plt.subplots(figsize=(7, 7))
ax.quiver(np.zeros(12), np.zeros(12), rotated[:, 0], rotated[:, 1],
          angles='xy', scale_units='xy', scale=1,
          color=plt.cm.viridis(np.linspace(0, 1, 12)),
          width=0.025, alpha=0.85)
circ = plt.Circle((0, 0), np.linalg.norm(cat), fill=False, linestyle='--', alpha=0.3)
ax.add_patch(circ)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.set_xlabel('dim 1'); ax.set_ylabel('dim 2')
ax.set_title("cat Rotated Through 12 Positions", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'rotation.png'), dpi=140, bbox_inches='tight')
plt.close()
print("✓ rotation.png")

# --- Section 3: Dot product heatmap ---
words = {'cat': cat, 'dog': dog, 'random': random_vec, 'neg_cat': -cat}
names = list(words.keys())
n = len(names)
scores = np.zeros((n, n))
for i, (ni, vi) in enumerate(words.items()):
    for j, (nj, vj) in enumerate(words.items()):
        scores[i, j] = np.dot(vi, vj)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(scores, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=11)
ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=11)
ax.set_title("Pairwise Dot Products (≡ QKᵀ Attention Scores)", fontsize=13)
plt.colorbar(im, ax=ax, label='Dot product')
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{scores[i,j]:.2f}', ha='center', va='center', fontsize=10, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'dot_heatmap.png'), dpi=140, bbox_inches='tight')
plt.close()
print("✓ dot_heatmap.png")

# --- Section 4: Normalized heatmap ---
cat_n = cat / np.linalg.norm(cat)
dog_n = dog / np.linalg.norm(dog)
rand_n = random_vec / np.linalg.norm(random_vec)
norm_words = {'cat': cat_n, 'dog': dog_n, 'random': rand_n, 'neg_cat': -cat_n}
norm_names = list(norm_words.keys())
n = len(norm_names)
scores_norm = np.zeros((n, n))
for i, (ni, vi) in enumerate(norm_words.items()):
    for j, (nj, vj) in enumerate(norm_words.items()):
        scores_norm[i, j] = np.dot(vi, vj)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(scores_norm, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_xticklabels(norm_names, fontsize=11)
ax.set_yticks(range(n)); ax.set_yticklabels(norm_names, fontsize=11)
ax.set_title("Normalized Dot Products — All Self-Similarities = 1.0", fontsize=13)
plt.colorbar(im, ax=ax, label='Cosine similarity')
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{scores_norm[i,j]:.2f}', ha='center', va='center', fontsize=10, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'norm_heatmap.png'), dpi=140, bbox_inches='tight')
plt.close()
print("✓ norm_heatmap.png")

# --- Section 6: 3D Project Up ---
W_up = np.array([[0.6, 0.1], [0.1, 0.7], [0.5, 0.5]])
v_3d = W_up @ cat

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(np.linspace(0, 0.85, 4), np.linspace(0, 0.85, 4))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.08, color='gray')
ax.quiver(0, 0, 0, cat[0], cat[1], 0,
          color='steelblue', linewidth=5, arrow_length_ratio=0.12, label='cat (2D, on floor)')
ax.quiver(0, 0, 0, v_3d[0], v_3d[1], v_3d[2],
          color='darkorange', linewidth=5, arrow_length_ratio=0.12, label='W_up @ cat (3D)')
ax.plot([v_3d[0], v_3d[0]], [v_3d[1], v_3d[1]], [0, v_3d[2]],
        '--', color='gray', alpha=0.5, linewidth=1.5)
ax.plot([v_3d[0], v_3d[0]], [0, v_3d[1]], [0, 0],
        ':', color='gray', alpha=0.3, linewidth=1)
ax.plot([0, v_3d[0]], [v_3d[1], v_3d[1]], [0, 0],
        ':', color='gray', alpha=0.3, linewidth=1)
ax.scatter([v_3d[0]], [v_3d[1]], [0], color='darkorange', alpha=0.5, s=80)
ax.set_xlim(0, 0.85); ax.set_ylim(0, 0.85); ax.set_zlim(0, 0.85)
ax.set_xlabel('dim 1 (x)', fontsize=10); ax.set_ylabel('dim 2 (y)', fontsize=10)
ax.set_zlabel('dim 3 (z)', fontsize=10)
ax.set_title('Projecting Up: 2D cat → 3D via W_up', fontsize=14)
ax.legend(loc='upper left', fontsize=9)
ax.view_init(elev=20, azim=-55)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'project_up.png'), dpi=140, bbox_inches='tight')
plt.close()
print("✓ project_up.png")

# --- Section 7: Round-trip ---
W_down = np.array([[0.4, 0.5, 0.2], [0.3, 0.1, 0.6]])
v_2d_back = W_down @ v_3d

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.08, color='gray')
ax1.quiver(0, 0, 0, cat[0], cat[1], 0,
           color='steelblue', linewidth=5, arrow_length_ratio=0.12, label='original cat (2D)')
ax1.quiver(0, 0, 0, v_3d[0], v_3d[1], v_3d[2],
           color='darkorange', linewidth=5, arrow_length_ratio=0.12, alpha=0.6, label='projected up (3D)')
ax1.quiver(0, 0, 0, v_2d_back[0], v_2d_back[1], 0,
           color='crimson', linewidth=5, arrow_length_ratio=0.12, label='back down (2D)')
ax1.plot([v_3d[0], v_3d[0]], [v_3d[1], v_3d[1]], [0, v_3d[2]],
         '--', color='gray', alpha=0.4, linewidth=1.5)
ax1.set_xlim(0, 0.85); ax1.set_ylim(0, 0.85); ax1.set_zlim(0, 0.85)
ax1.set_xlabel('dim 1'); ax1.set_ylabel('dim 2'); ax1.set_zlabel('dim 3')
ax1.set_title('3D Round-Trip', fontsize=13)
ax1.legend(loc='upper left', fontsize=8)
ax1.view_init(elev=20, azim=-55)

ax2 = fig.add_subplot(1, 2, 2)
ax2.quiver(0, 0, cat[0], cat[1], angles='xy', scale_units='xy', scale=1,
          color='steelblue', width=0.05, label='original cat', zorder=3)
ax2.quiver(0, 0, v_2d_back[0], v_2d_back[1], angles='xy', scale_units='xy', scale=1,
          color='crimson', width=0.05, label='after round-trip', zorder=3)
ax2.plot([cat[0], v_2d_back[0]], [cat[1], v_2d_back[1]],
         'k--', alpha=0.4, linewidth=2, label=f'|Δ| = {np.linalg.norm(v_2d_back - cat):.3f}')
ax2.set_xlim(0, 0.85); ax2.set_ylim(0, 0.85)
ax2.set_xlabel('dim 1'); ax2.set_ylabel('dim 2')
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
ax2.set_title('Original vs Round-Trip (2D)', fontsize=13)
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'round_trip.png'), dpi=140, bbox_inches='tight')
plt.close()
print("✓ round_trip.png")

print(f"\nAll images saved to {OUT}/")
