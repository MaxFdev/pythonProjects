import numpy as np
import matplotlib.pyplot as plt

"""
Generating graph
"""

def generate_random_graph(n, p):
    # Initialize an n-by-n adjacency matrix with zeros
    G = np.zeros((n, n), dtype=int)  

    # Iterate over each pair of nodes 
    for i in range(n):
        for j in range(i+1, n):

            # puts an edge with probability p (if random is < p)
            if np.random.random() < p:
                G[i, j] = 1
                G[j, i] = 1

    return G

"""
Running simulation
"""

def simulate_epidemic(G, seed, t, p):
    # Number of individuals
    n = len(G)

    # Initialize the sets S, I, and R
    
    # Susceptible individuals
    S = set(range(n))  
    
    # Infected individuals
    I = set(seed)  
    
    # Recovered individuals
    R = set()  

    # Run the simulation until nobody has the infection
    while I:
        
        new_infections = set()
        
        # Infected individuals to be removed from I
        to_remove = set()  

        # Iterate over infected individuals
        for infected in I:

            # Find neighbors of infected individual
            neighbors = np.nonzero(G[infected])[0]

            # Find susceptible neighbors
            susceptible_neighbors = []

            # Iterate over each neighbor in the neighbors list
            for neighbor in neighbors:

                # Check if the neighbor is in the set S
                if neighbor in S:

                    # If the neighbor is susceptible, add it to the susceptible_neighbors list
                    susceptible_neighbors.append(neighbor)

            # Transmission to susceptible neighbors with probability p
            for neighbor in susceptible_neighbors:
                if np.random.random() < p:
                    new_infections.add(neighbor)

            # Move infected individual to R after t periods
            if infected not in new_infections:
                to_remove.add(infected)
                R.add(infected)

        # Remove infected individuals from I
        I.difference_update(to_remove)

        # Move newly infected individuals to I
        I.update(new_infections)

        # Move individuals in I to R after t periods
        completed_infections = []
        for infected in I:
            if np.random.random() < (1 / t):
                completed_infections.append(infected)
        I.difference_update(completed_infections)
        R.update(completed_infections)
        S.difference_update(completed_infections)

    return len(S), len(R)

"""
Test
"""

def run_test(p_v):
    # Set the provided p_values
    p_values = p_v  
    
    # Number of iterations for each combination of p1 and p2
    num_iterations = 100  
    
    # Dictionary to store the results
    results = {}  

    # Iterate over p_values
    for p1 in p_values:
        for p2 in p_values:
            
            # Total count of uninfected individuals
            uninfected_total = 0  
            
            # Total count of infected individuals
            infected_total = 0  

            # Run simulations for each combination of p1 and p2
            for _ in range(num_iterations):
                
                # Generate a random graph
                G = generate_random_graph(10, p1)  
                
                # Simulate epidemic and get counts
                uninfected, infected = simulate_epidemic(G, {1}, 1, p2)

                # Accumulate uninfected count
                uninfected_total += uninfected  

                # Accumulate infected count
                infected_total += infected 

            # Calculate average counts
            avg_uninfected = uninfected_total / num_iterations
            avg_infected = infected_total / num_iterations

            # Store the results in the dictionary
            results[(p1, p2)] = (avg_uninfected, avg_infected)

    return results


p_values = np.arange(0, 1.1, 0.1)
results = run_test(p_values)

"""
Plot the results (Tested a few ways to display the graph)
"""

"""
Option1
"""

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# # Define X and Y for contour plotting
# X, Y = np.meshgrid(p_values, p_values)

# # Define Z_S and Z_R for the average sizes of S and R
# Z_S = np.array([[results[(p1, p2)][0] for p2 in p_values] for p1 in p_values])
# Z_R = np.array([[results[(p1, p2)][1] for p2 in p_values] for p1 in p_values])

# # Plot average size of susceptible individuals
# contour_S = ax1.contourf(X, Y, Z_S, levels=10, cmap='viridis')
# ax1.set_title('Average Size of S')
# ax1.set_xlabel('p1')
# ax1.set_ylabel('p2')

# # Plot average size of removed individuals
# contour_R = ax2.contourf(X, Y, Z_R, levels=10, cmap='viridis')
# ax2.set_title('Average Size of R')
# ax2.set_xlabel('p1')
# ax2.set_ylabel('p2')

# # Add colorbar using the mappable objects
# fig.colorbar(contour_R, ax=ax2)

# plt.tight_layout()
# plt.show()

"""
Option2
"""

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# # Define X for the x-axis
# X = p_values

# # Define Y_S and Y_R for the average sizes of S and R
# Y_S = [results[(p, p)][0] for p in p_values]
# Y_R = [results[(p, p)][1] for p in p_values]

# # Plot average size of susceptible individuals
# ax1.plot(X, Y_S, marker='o', linestyle='-')
# ax1.set_title('Average Size of S')
# ax1.set_xlabel('p1')
# ax1.set_ylabel('p2')

# # Plot average size of removed individuals
# ax2.plot(X, Y_R, marker='o', linestyle='-')
# ax2.set_title('Average Size of R')
# ax2.set_xlabel('p1')
# ax2.set_ylabel('p2')

# plt.tight_layout()
# plt.show()

"""
Option3
"""

# # Extract the X, Y, and Z values from the results dictionary
# X = p_values
# Y = p_values
# Z_S = np.array([[results[(p1, p2)][0] for p2 in p_values] for p1 in p_values])
# Z_R = np.array([[results[(p1, p2)][1] for p2 in p_values] for p1 in p_values])

# # Extract the Y_S and Y_R values from the results dictionary
# Y_S = np.array([[results[(p1, p2)][0] for p2 in p_values] for p1 in p_values])
# Y_R = np.array([[results[(p1, p2)][1] for p2 in p_values] for p1 in p_values])

# # Create subplots for the line plot and two heat maps
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# # Plot the line plot of average size of susceptible individuals
# ax1.plot(X, Y_S, marker='o', linestyle='-')
# ax1.set_title('Average Size of S')
# ax1.set_xlabel('p1')
# ax1.set_ylabel('p2')

# # Plot the line plot of average size of removed individuals
# ax2.plot(X, Y_R, marker='o', linestyle='-')
# ax2.set_title('Average Size of R')
# ax2.set_xlabel('p1')
# ax2.set_ylabel('p2')

# # Create a heat map for the average size of susceptible individuals
# heatmap_S = ax3.imshow(Z_S, cmap='hot', extent=[0, 1, 0, 1], origin='lower')
# ax3.set_title('Heatmap of Average Size of S')
# ax3.set_xlabel('p1')
# ax3.set_ylabel('p2')
# fig.colorbar(heatmap_S, ax=ax3, label='Average Size')

# # Create a heat map for the average size of removed individuals
# heatmap_R = ax4.imshow(Z_R, cmap='hot', extent=[0, 1, 0, 1], origin='lower')
# ax4.set_title('Heatmap of Average Size of R')
# ax4.set_xlabel('p1')
# ax4.set_ylabel('p2')
# fig.colorbar(heatmap_R, ax=ax4, label='Average Size')

# plt.tight_layout()
# plt.show()