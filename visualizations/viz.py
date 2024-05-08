import matplotlib.pyplot as plt
import pandas as pd

# plots
fig, ax = plt.subplots()
# Load the data
df1 = pd.read_csv("data/data.csv")  
# Extract the step and episode_reward columns
steps = df1['Step'].values
loss = df1['Loss'].values
returns= df1['Return'].values
return_stds = df1['Return Std'].values
loss_stds = df1['Loss Std'].values


# calculate the upper and lower bound of the returns and loss
up1 = [returns[i] + return_stds[i] for i in range(len(returns))]
down1 = [returns[i] - return_stds[i] for i in range(len(returns))]
up2 = [loss[i] + loss_stds[i] for i in range(len(loss))]
down2 = [loss[i] - loss_stds[i] for i in range(len(loss))]
# plot the data
ax.plot(steps, returns, label='TOLD-ZERO MCTS, Cartpole_v1, returns', color='blue')
ax.plot(steps, loss, label='TOLD-ZERO MCTS, Cartpole_v1, loss', color='red')
# plot the upper and lower bound
ax.fill_between(steps, down1, up1, alpha=0.2)
ax.fill_between(steps, down2, up2, alpha=0.2)
# set the title and labels
ax.legend()
ax.set_xlabel('Steps')
ax.set_ylabel('Loss & Returns')
plt.grid(True, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
plt.savefig('outputs/cartpole_returns.png') 