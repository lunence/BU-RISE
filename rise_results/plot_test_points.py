import matplotlib.pyplot as plt

# Points to plot
points = [
    (0.0, 0.0),
    (0.5, 0.5),
    (-0.5, 0.5),
    (-0.25, -0.25),
    (0.25, -0.25),
]

x = [p[0] for p in points]
y = [p[1] for p in points]

plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=100)

# Label each point
for i, (px, py) in enumerate(points):
    plt.text(px + 0.02, py + 0.02, f"({px}, {py})", fontsize=9)

# Draw x and y axes
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

plt.xlim(-0.75, 0.75)
plt.ylim(-0.5, 0.75)
plt.gca().set_aspect("equal", adjustable="box")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Navigation Goal Locations")
plt.grid(True)

plt.tight_layout()
plt.show()