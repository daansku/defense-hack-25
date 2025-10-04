import matplotlib.pyplot as plt
import numpy as np

plt.ion()  # Turn on interactive mode

for i in range(10):
    plt.clf()  # Clear the figure
    plt.plot(np.random.rand(10))
    plt.pause(0.5)  # Pause for 0.5 seconds, keeps plot responsive
    print(f"Iteration {i}")

plt.ioff()  # Turn off interactive mode
plt.show()