import numpy as np

def dp(dist_mat):
    N, M = dist_mat.shape

    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]

    # Define the region for traceback (modify this part)
    min_len = min(N, M)
    while i > 0 or j > 0:
        if len(path) == min_len:
            break  # Stop when the path length reaches the minimum length
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    
    # Adjust the cost matrix to include only the new path
    alignment_cost = 0
    for i, j in path:
        alignment_cost += dist_mat[i, j]
    
    return (path[::-1], alignment_cost)

if __name__ == "__main__":
    idx = np.linspace(0, 6.28, num=100)
    x = np.sin(idx) + np.random.uniform(size=100) / 10.0  # query
    y = np.cos(idx)  # reference
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
    plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
    plt.axis("off")

    # Distance matrix
    N = x.shape[0]
    M = y.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = abs(x[i] - y[j])

    # DTW
    path, alignment_cost = dp(dist_mat)
    print("Alignment cost: {:.4f}".format(alignment_cost))

    plt.figure(figsize=(6, 4))
    plt.subplot(121)
    plt.title("Distance matrix")
    plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    plt.subplot(122)
    plt.title("Cost matrix")
    plt.imshow(np.array([[alignment_cost]]), cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    x_path, y_path = zip(*path)
    plt.plot(y_path, x_path)

    plt.figure()
    for x_i, y_j in path:
        plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="C7")
    plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
    plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
    plt.axis("off")

    plt.show()
