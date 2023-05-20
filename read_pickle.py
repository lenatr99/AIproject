import pickle
import matplotlib.pyplot as plt 

with open('scores.pickle', 'rb') as file:
    data = pickle.load(file)


# Use the 'data' object as per your requirements

# get average of every 5 scores
chunk_size = 1
averages = []
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    average = sum(chunk) / len(chunk)
    averages.append(average)

print(averages)
fig, ax = plt.subplots() 
x_values = list(range(len(averages)))  # X-axis represents the number of episodes
ax.plot(x_values, averages)
ax.set_xlabel('Episode')
ax.set_ylabel('Score')
ax.set_title('Scores over Episodes')
plt.pause(0.001)  # Pause to allow the plot to update
plt.show()
# Close the file (not required if using 'with' statement)
file.close()